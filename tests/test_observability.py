"""Tracking is optional and backend-agnostic: no env -> no run; the run handle
fans metrics/params out to whatever backends are live and never raises."""
import sys

from lmkit import observability
from lmkit.observability import _Run, make_run


def test_make_run_none_without_any_backend(monkeypatch):
    monkeypatch.delenv("AIM_REPO", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert make_run("exp", hparams={"lr": 1e-3}) is None


def test_make_mlflow_none_when_uri_unset(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert observability._make_mlflow("exp", None, None, None) is None


class _FakeAim:
    def __init__(self):
        self.metrics, self.items, self.closed = [], {}, False

    def track(self, value, name, step=None):
        self.metrics.append((name, value, step))

    def __setitem__(self, k, v):
        self.items[k] = v

    def close(self):
        self.closed = True


class _FakeMlflowClient:
    def __init__(self):
        self.metrics, self.params, self.terminated = [], {}, None
        self.status = None

    def log_metric(self, run_id, key, value, step=0):
        self.metrics.append((run_id, key, value, step))

    def log_param(self, run_id, key, value):
        self.params[key] = value

    def set_terminated(self, run_id, status="FINISHED"):
        self.terminated = run_id
        self.status = status


def test_run_fans_out_to_both_backends():
    aim, mlf = _FakeAim(), _FakeMlflowClient()
    run = _Run()
    run._aim, run._mlflow = aim, (mlf, "rid")

    assert run  # truthy with a backend live
    run.track(0.5, name="loss", step=3)
    run["hparams"] = {"lr": 0.01, "layers": 4}
    run.close()

    assert aim.metrics == [("loss", 0.5, 3)]
    assert ("rid", "loss", 0.5, 3) in mlf.metrics
    assert mlf.params == {"lr": "0.01", "layers": "4"}
    assert aim.closed and mlf.terminated == "rid"
    assert mlf.status == "FINISHED"  # default terminal status
    assert not run  # both cleared after close


def test_close_passes_terminal_status():
    mlf = _FakeMlflowClient()
    run = _Run()
    run._mlflow = (mlf, "rid")
    run.close("KILLED")
    assert mlf.status == "KILLED"


def test_canonical_hparams_renames_and_tags():
    hp = observability.canonical_hparams(
        "lmkit", 7, {"n_layer": 12, "n_head": 12, "batch_size": 2})
    assert hp == {"framework": "lmkit", "params": 7,
                  "n_layers": 12, "n_heads": 12, "batch_size": 2}


def test_empty_run_is_falsy_and_inert():
    run = _Run()
    assert not run
    run.track(1.0, name="x", step=0)  # must not raise with no backends
    run.close()


def test_run_id_sidecar_roundtrip(tmp_path):
    assert observability.stored_run_id(tmp_path) is None
    observability.store_run_id(tmp_path, "abc123")
    assert observability.stored_run_id(tmp_path) == "abc123"
    observability.store_run_id(tmp_path, None)  # no id -> no write, no raise
    assert observability.stored_run_id(tmp_path) == "abc123"


class _ReattachClient:
    """MlflowClient fake for the resume path: knows one existing run."""
    existing = "run-live"

    def __init__(self):
        _ReattachClient.last = self
        self.updated, self.created = None, False

    def get_run(self, run_id):
        if run_id != self.existing:
            raise KeyError(run_id)

    def update_run(self, run_id, status=None):
        self.updated = (run_id, status)

    def get_experiment_by_name(self, name):
        return type("E", (), {"experiment_id": "1"})()

    def create_run(self, exp_id, tags=None):
        self.created = True
        return type("R", (), {"info": type("I", (), {"run_id": "run-new"})()})()

    def log_param(self, run_id, k, v):
        pass


def _fake_mlflow(monkeypatch):
    import types
    tracking = types.ModuleType("mlflow.tracking")
    tracking.MlflowClient = _ReattachClient
    mlflow = types.ModuleType("mlflow")
    mlflow.tracking = tracking
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake")


def test_make_mlflow_reattaches_to_live_run(monkeypatch):
    _fake_mlflow(monkeypatch)
    got = observability._make_mlflow("exp", None, None, None, "run-live")
    client, run_id = got
    assert run_id == "run-live"
    assert client.updated == ("run-live", "RUNNING")
    assert not client.created  # reattach must not open a second run


def test_make_mlflow_falls_back_when_run_is_gone(monkeypatch):
    _fake_mlflow(monkeypatch)
    got = observability._make_mlflow("exp", None, None, None, "run-gone")
    _, run_id = got
    assert run_id == "run-new"
    assert _ReattachClient.last.created
