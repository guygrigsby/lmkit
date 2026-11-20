# Cloud job image

A job is one shell command run in a container on a cloud GPU pod. Everything the job needs
is in the image or its environment, and everything it produces is uploaded before the pod
stops. See `docs/adr/0001-cloud-jobs-as-container-images.md` for why.

## Contract

Environment, read by `lmkit.cloudjob`:

| Variable | Meaning | Default |
|---|---|---|
| `LMKIT_JOB_NAME` | run id; the folder the outputs land in | required |
| `LMKIT_JOB_CMD` | the shell command to run | required |
| `LMKIT_OUT_REPO` | hub repo id, created private if missing; a public repo aborts the job | required |
| `LMKIT_OUT_DIR` | directory the job writes outputs to; exported to the command | `/out` |
| `LMKIT_MAX_SECONDS` | hard time limit; the process group gets SIGTERM, then SIGKILL | 21600 |
| `HF_TOKEN` | hub write token | none |

Outputs uploaded to `<LMKIT_OUT_REPO>/<LMKIT_JOB_NAME>/`: everything under `LMKIT_OUT_DIR`,
plus `job.log` (combined stdout and stderr) and `status.txt` (outcome and elapsed time).
Upload happens after success, failure and timeout alike, with four attempts and exponential
backoff.

Exit code: the job's own; 124 on timeout; 70 when the upload failed; 2 when required
configuration is missing.

## Lifecycle

1. Pod starts from the image. The entrypoint checks configuration and that the output repo is
   private, then looks for `<LMKIT_JOB_NAME>/status.txt` in the repo. If it exists the job
   already finished: nothing runs, and the entrypoint goes straight to step 4.
2. The command runs. SIGTERM to the container (pod stop) terminates the command and still
   uploads.
3. Outputs are uploaded (four attempts, exponential backoff).
4. On a pod (`RUNPOD_POD_ID` set), the entrypoint deletes the pod through the platform API
   with the `RUNPOD_API_KEY` the platform injects. Exiting is not enough: the platform
   restarts an exited container, which would rerun the job, so a finished container that
   cannot delete its pod holds (sleeps) instead of exiting.

The marker makes restarts and repeat launches of a finished job harmless. It also means a
failed job is not retried automatically: relaunch it under a new name.

## Image

`docker/cuda.Dockerfile`: the RunPod PyTorch 2.8 / CUDA 12.8 / Python 3.12 base, plus pinned
`transformers`, `peft`, `flash-attn` and this repository's package. CI builds it for
`linux/amd64` on every push (`make image`) and pushes `latest` and the commit sha from
`main`. Weights are downloaded by the job command into `HF_HOME` (`/cache/hf`) at start.

## Failure modes

| Failure | Behavior |
|---|---|
| Command fails | exit code recorded, log and partial outputs uploaded |
| Command hangs | killed at `LMKIT_MAX_SECONDS`, exit 124, outputs uploaded |
| Pod preempted or stopped | SIGTERM path uploads what exists; a restarted container sees the marker (if the job finished) or reruns it |
| Hub unreachable at the end | retried; then the pod is held, not deleted, so the outputs are not lost; exit 70 is only reached off the platform |
| Pod deletion fails | the container holds; a later sweep or the launcher removes the pod |
| Container restarted after finishing | marker found, job skipped, pod deleted |
| Output repo is public | job refused before running |

## Not covered

Scheduling and launching (a launcher script lives with the project that uses this),
multi-node training, and a ROCm image for local machines.
