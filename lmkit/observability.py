"""Optional experiment tracking.

Fans out to any configured backend; no-ops otherwise, so training never depends
on a tracker being installed or reachable. Endpoints come from the environment,
never baked in:

  - Aim:    ``AIM_REPO``            (e.g. ``aim://host:53800``)
  - MLflow: ``MLFLOW_TRACKING_URI`` (e.g. ``http://bee:5000``)

Both may be set; metrics then go to both. On the GPU fleet, ``gputex`` injects
``MLFLOW_TRACKING_URI`` into every job, so a run launched through the GPU gate is
tracked by construction. ``make_run`` returns a run handle, or ``None`` when no
backend is configured/reachable. Callers log with
``run.track(value, name=..., step=...)`` and call ``run.close()`` at the end;
guard every use with ``if run:``.
"""
from __future__ import annotations

import os
import sys
from typing import Optional


def _warn(msg: str) -> None:
    print(f"[lmkit.observability] {msg}", file=sys.stderr)


# Fleet-canonical hparam names. Every framework (lmkit, lmkit-go, lmtk) logs the
# same concept under the same key so params line up across frameworks in the
# tracker. Maps each framework's local spelling to the canonical one; keys not
# listed are already canonical.
_CANONICAL_KEYS = {
    "n_layer": "n_layers",
    "n_head": "n_heads",
    "layers": "n_layers",
    "heads": "n_heads",
    "kv_heads": "n_kv_heads",
    "vocab": "vocab_size",
    "hidden": "hidden_size",
    "inter": "intermediate_size",
    "ffn_hidden": "intermediate_size",
    "block": "block_size",
    "seq_len": "block_size",
    "batch": "batch_size",
    "warmup": "warmup_steps",
}


def canonical_hparams(framework: str, n_params: int, *cfgs) -> dict:
    """Merge config dicts into one hparam dict under the fleet-canonical key
    names, tagged with the emitting framework."""
    out = {"framework": framework, "params": n_params}
    for cfg in cfgs:
        d = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)
        for k, v in d.items():
            out[_CANONICAL_KEYS.get(k, k)] = v
    return out


def _short(v) -> str:
    """Coerce a param value to a short string (MLflow rejects long params)."""
    s = str(v)
    return s if len(s) <= 250 else s[:247] + "..."


class _Run:
    """A run handle that fans ``.track`` / ``.close`` / ``[]=`` out to every
    active backend. Truthy only when at least one backend is live, so the
    caller's ``if run:`` guard keeps working unchanged.
    """

    def __init__(self):
        self._aim = None     # aim.Run, or None
        self._mlflow = None  # (MlflowClient, run_id), or None

    def __bool__(self) -> bool:
        return self._aim is not None or self._mlflow is not None

    def track(self, value, name: str, step: Optional[int] = None) -> None:
        if self._aim is not None:
            try:
                self._aim.track(value, name=name, step=step)
            except Exception as e:
                _warn(f"Aim track failed: {e}")
        if self._mlflow is not None:
            client, run_id = self._mlflow
            try:
                client.log_metric(run_id, name, float(value), step=step or 0)
            except Exception as e:
                _warn(f"MLflow track failed: {e}")

    def __setitem__(self, key, value) -> None:
        if self._aim is not None:
            try:
                self._aim[key] = value
            except Exception:
                pass
        if self._mlflow is not None:
            client, run_id = self._mlflow
            try:
                items = value.items() if isinstance(value, dict) else [(key, value)]
                for k, v in items:
                    client.log_param(run_id, str(k), _short(v))
            except Exception as e:
                _warn(f"MLflow param failed: {e}")

    def close(self, status: str = "FINISHED") -> None:
        """End the run. ``status`` is an MLflow terminal status — FINISHED for a
        completed run, FAILED for divergence (non-finite loss), KILLED for a
        SIGTERM stop — so the tracker distinguishes how runs ended. Aim has no
        status concept; it just closes."""
        if self._aim is not None:
            try:
                self._aim.close()
            except Exception:
                pass
            self._aim = None
        if self._mlflow is not None:
            client, run_id = self._mlflow
            try:
                client.set_terminated(run_id, status=status)
            except Exception:
                pass
            self._mlflow = None


def make_run(experiment: str, hparams: Optional[dict] = None,
             name: Optional[str] = None, description: Optional[str] = None):
    """Start a tracking run on every configured backend (Aim, MLflow). Returns a
    run handle, or ``None`` if no backend is configured/reachable.

    ``experiment`` groups runs; ``name`` is the human label shown in the runs
    list; ``description`` is a one-line summary; ``hparams`` are recorded as
    params. The caller logs with ``run.track(value, name=..., step=...)`` and
    calls ``run.close()``; guard every use with ``if run:``.
    """
    run = _Run()
    run._aim = _make_aim(experiment, hparams, name, description)
    run._mlflow = _make_mlflow(experiment, hparams, name, description)
    return run if run else None


def _make_aim(experiment, hparams, name, description):
    repo = os.environ.get("AIM_REPO")
    if not repo:
        return None
    try:
        import aim
    except Exception:
        # AIM_REPO says the caller expects tracking; a missing client must be
        # loud or runs silently vanish from the tracker.
        _warn("Aim tracking disabled: AIM_REPO is set but 'aim' is not installed")
        return None
    try:
        run = aim.Run(repo=repo, experiment=experiment)
        if name:
            run.name = name
        if description:
            run.description = description
        if hparams:
            run["hparams"] = dict(hparams)
        return run
    except Exception as e:  # unreachable server, version skew, etc.
        _warn(f"Aim tracking disabled: {e}")
        return None


def _make_mlflow(experiment, hparams, name, description):
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        return None
    try:
        from mlflow.tracking import MlflowClient
    except Exception:
        # MLFLOW_TRACKING_URI says the caller expects tracking; a missing
        # client must be loud or runs silently vanish from the tracker.
        _warn("MLflow tracking disabled: MLFLOW_TRACKING_URI is set but "
              "'mlflow' is not installed (pip install mlflow-skinny)")
        return None
    try:
        client = MlflowClient()  # reads MLFLOW_TRACKING_URI from the environment
        exp = client.get_experiment_by_name(experiment)
        exp_id = exp.experiment_id if exp else client.create_experiment(experiment)
        # Set name/description via tags so this works across MLflow versions
        # (the run_name kwarg is newer than the underlying tag).
        tags = {}
        if name:
            tags["mlflow.runName"] = name
        if description:
            tags["mlflow.note.content"] = description
        run = client.create_run(exp_id, tags=tags or None)
        run_id = run.info.run_id
        if hparams:
            for k, v in dict(hparams).items():
                try:
                    client.log_param(run_id, str(k), _short(v))
                except Exception:
                    pass  # one bad param must not abort the run
        return (client, run_id)
    except Exception as e:  # unreachable server, version skew, etc.
        _warn(f"MLflow tracking disabled: {e}")
        return None
