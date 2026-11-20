"""Run one job inside a container, ship its outputs to a private hub repo, exit.

The container entrypoint for cloud GPU pods: nobody stays connected. The job is one
shell command; everything it writes under $LMKIT_OUT_DIR, plus its combined output
(`job.log`) and an outcome line (`status.txt`), is uploaded to `LMKIT_OUT_REPO` under
`LMKIT_JOB_NAME/`, even when the job fails or times out. The process then exits with
the job's code, which stops the pod, so it needs no credentials to delete itself.

    LMKIT_JOB_NAME   run id, the folder in the repo (required)
    LMKIT_JOB_CMD    shell command to run (required)
    LMKIT_OUT_REPO   hub repo id, created private if missing (required)
    LMKIT_OUT_DIR    where the job writes outputs (default /out)
    LMKIT_MAX_SECONDS  hard limit on the job (default 21600)
    HF_TOKEN         hub write token

Exit code: the job's, 124 on timeout, 70 when outputs could not be uploaded.

On a RunPod pod (RUNPOD_POD_ID set) the pod is deleted once the outputs are safely in the
repo: the platform restarts an exited container, so exiting alone would rerun the job.
The platform injects RUNPOD_POD_ID and RUNPOD_API_KEY into the container. If the upload
or the deletion fails the process holds the pod instead of exiting, because exiting would
destroy the only copy of the results or restart the job. A job whose `status.txt` is
already in the repo is not run again, so a restart (or a launch of a finished job) is a
no-op that goes straight to termination.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from lmkit.push import ensure_repo

TIMEOUT = 124
UPLOAD_FAILED = 70
_GRACE = 20  # seconds between SIGTERM and SIGKILL


def _hub():
    from huggingface_hub import HfApi

    return HfApi(token=os.environ.get("HF_TOKEN"))


def _kill_group(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=_GRACE if sig == signal.SIGTERM else 5)
            return
        except subprocess.TimeoutExpired:
            continue


def _execute(cmd: str, out: Path, max_seconds: int) -> tuple[int, str]:
    """Run `cmd`, teeing its output to stdout and out/job.log. Returns (rc, outcome)."""
    env = {**os.environ, "LMKIT_OUT_DIR": str(out)}
    proc = subprocess.Popen(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    stopped = threading.Event()
    old = signal.signal(signal.SIGTERM, lambda *_: (stopped.set(), _kill_group(proc)))

    def pump():
        with open(out / "job.log", "wb") as log:
            for line in proc.stdout:
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
                log.write(line)

    t = threading.Thread(target=pump)
    t.start()
    started = time.monotonic()
    try:
        proc.wait(timeout=max_seconds)
        outcome = f"exit {proc.returncode}"
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        rc, outcome = TIMEOUT, f"timeout after {max_seconds}s"
    finally:
        signal.signal(signal.SIGTERM, old)
    t.join()
    if stopped.is_set():
        outcome = f"terminated by SIGTERM ({outcome})"
    (out / "status.txt").write_text(f"{outcome}\nelapsed {time.monotonic() - started:.0f}s\n")
    return rc, outcome


def _upload(api, out: Path, repo: str, name: str, retries: int, sleep) -> bool:
    for attempt in range(retries):
        try:
            api.upload_folder(folder_path=str(out), path_in_repo=name, repo_id=repo, repo_type="model")
            return True
        except Exception as e:  # transient network/hub errors: retry with backoff
            print(f"cloudjob: upload attempt {attempt + 1}/{retries} failed: {e}", file=sys.stderr)
            sleep(2 ** attempt * 5)
    return False


def run_job(name: str, cmd: str, out_dir, repo: str, max_seconds: int, *, api=None,
            retries: int = 4, sleep=None) -> int:
    api = api or _hub()
    sleep = sleep or time.sleep
    ensure_repo(api, repo, private=True)  # refuse before spending GPU time on a public repo
    if api.file_exists(repo, f"{name}/status.txt", repo_type="model"):
        print(f"cloudjob: {name} already finished (status.txt in {repo}); not running it again")
        return 0
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rc, _ = _execute(cmd, out, max_seconds)
    if not _upload(api, out, repo, name, retries, sleep):
        print("cloudjob: outputs NOT uploaded; they remain in the container only", file=sys.stderr)
        return UPLOAD_FAILED
    return rc


def _terminate_pod() -> bool:
    """Delete this pod through the platform API using the injected credentials."""
    pod, key = os.environ.get("RUNPOD_POD_ID"), os.environ.get("RUNPOD_API_KEY")
    if not (pod and key):
        print("cloudjob: RUNPOD_POD_ID/RUNPOD_API_KEY not set; cannot terminate", file=sys.stderr)
        return False
    for attempt in range(3):
        try:
            req = urllib.request.Request(f"https://api.runpod.io/v2/pods/{pod}", method="DELETE",
                                         headers={"Authorization": f"Bearer {key}",
                                                  "User-Agent": "lmkit-cloudjob"})
            with urllib.request.urlopen(req, timeout=30):
                return True
        except Exception as e:  # platform API blips: retry with backoff
            print(f"cloudjob: terminate attempt {attempt + 1}/3 failed: {e}", file=sys.stderr)
            time.sleep(2 ** attempt * 3)
    return False


def _hold(why: str) -> None:
    """Keep the container alive: exiting would restart the job or lose the outputs."""
    print(f"cloudjob: holding the pod: {why}", file=sys.stderr, flush=True)
    while True:
        time.sleep(3600)


def main() -> int:
    need = ("LMKIT_JOB_NAME", "LMKIT_JOB_CMD", "LMKIT_OUT_REPO")
    missing = [k for k in need if not os.environ.get(k)]
    if missing:
        print(f"cloudjob: missing {', '.join(missing)}", file=sys.stderr)
        return 2
    rc = run_job(os.environ["LMKIT_JOB_NAME"], os.environ["LMKIT_JOB_CMD"],
                 os.environ.get("LMKIT_OUT_DIR", "/out"), os.environ["LMKIT_OUT_REPO"],
                 int(os.environ.get("LMKIT_MAX_SECONDS", "21600")), api=_hub())
    if os.environ.get("RUNPOD_POD_ID"):
        if rc == UPLOAD_FAILED:
            _hold("outputs were not uploaded and exist only in this container")
        elif not _terminate_pod():
            _hold("could not terminate the pod")
    return rc


if __name__ == "__main__":
    sys.exit(main())
