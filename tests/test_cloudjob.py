"""cloudjob: run one command in a container, ship /out to a private hub repo, exit.
The hub is a fake with the three HfApi calls lmkit.push uses; the command is a real
subprocess, so timeouts and exit codes are exercised as in production."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lmkit import cloudjob


class FakeApi:
    def __init__(self, private=True, fail_uploads=0):
        self.private, self.fail_uploads = private, fail_uploads
        self.uploads, self.created = [], []

    def create_repo(self, repo_id, **kw):
        self.created.append((repo_id, kw))

    def repo_info(self, repo_id, **kw):
        return SimpleNamespace(private=self.private)

    def upload_folder(self, **kw):
        if self.fail_uploads:
            self.fail_uploads -= 1
            raise ConnectionError("flaky")
        # snapshot what would be shipped: the hub copies at call time
        kw["files"] = sorted(p.name for p in Path(kw["folder_path"]).iterdir())
        self.uploads.append(kw)


def run(tmp_path, cmd, api=None, max_seconds=30, **kw):
    api = api or FakeApi()
    rc = cloudjob.run_job("job1", cmd, tmp_path / "out", "me/runs", max_seconds,
                          api=api, sleep=lambda s: None, **kw)
    return rc, api


def test_success_ships_outputs_and_log_to_a_private_repo(tmp_path):
    rc, api = run(tmp_path, 'echo hello && echo data > "$LMKIT_OUT_DIR/result.txt"')
    assert rc == 0
    (up,) = api.uploads
    assert up["repo_id"] == "me/runs" and up["path_in_repo"] == "job1"
    assert {"result.txt", "job.log", "status.txt"} <= set(up["files"])
    assert api.created[0][1]["private"] is True
    assert "hello" in (tmp_path / "out" / "job.log").read_text()


def test_failure_exit_code_propagates_and_outputs_still_ship(tmp_path):
    rc, api = run(tmp_path, "echo boom >&2; exit 3")
    assert rc == 3
    assert len(api.uploads) == 1
    assert "boom" in (tmp_path / "out" / "job.log").read_text()
    assert "exit 3" in (tmp_path / "out" / "status.txt").read_text()


def test_timeout_kills_the_whole_process_tree_and_still_ships(tmp_path):
    rc, api = run(tmp_path, "sleep 60 & sleep 60", max_seconds=1)
    assert rc == cloudjob.TIMEOUT
    assert len(api.uploads) == 1
    assert "timeout" in (tmp_path / "out" / "status.txt").read_text()


def test_upload_is_retried_with_backoff(tmp_path):
    rc, api = run(tmp_path, "true", api=FakeApi(fail_uploads=2))
    assert rc == 0 and len(api.uploads) == 1


def test_upload_that_never_succeeds_is_its_own_exit_code(tmp_path):
    rc, api = run(tmp_path, "true", api=FakeApi(fail_uploads=99))
    assert rc == cloudjob.UPLOAD_FAILED


def test_a_public_repo_is_refused_before_anything_is_shipped(tmp_path):
    api = FakeApi(private=False)
    with pytest.raises(RuntimeError, match="NOT private"):
        run(tmp_path, "true", api=api)
    assert api.uploads == []


def test_main_reads_the_job_from_the_environment(tmp_path, monkeypatch):
    api = FakeApi()
    monkeypatch.setattr(cloudjob, "_hub", lambda: api)
    monkeypatch.setenv("LMKIT_JOB_NAME", "j")
    monkeypatch.setenv("LMKIT_JOB_CMD", f'{sys.executable} -c "print(1)"')
    monkeypatch.setenv("LMKIT_OUT_REPO", "me/runs")
    monkeypatch.setenv("LMKIT_OUT_DIR", str(tmp_path / "o"))
    assert cloudjob.main() == 0
    assert api.uploads[0]["path_in_repo"] == "j"


def test_main_reports_missing_configuration_instead_of_running(monkeypatch, capsys):
    for k in ("LMKIT_JOB_NAME", "LMKIT_JOB_CMD", "LMKIT_OUT_REPO"):
        monkeypatch.delenv(k, raising=False)
    assert cloudjob.main() == 2
    assert "LMKIT_JOB_CMD" in capsys.readouterr().err
