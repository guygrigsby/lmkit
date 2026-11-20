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

Exit code: the job's own; 124 on timeout; 70 when the upload failed (outputs are then lost
with the pod); 2 when required configuration is missing.

## Lifecycle

1. Pod starts from the image; the entrypoint checks configuration and that the output repo
   is private.
2. The command runs. SIGTERM to the container (pod stop) terminates the command and still
   uploads.
3. The entrypoint exits, which stops the pod. The entrypoint holds no cloud-account
   credentials, so it cannot delete the pod; a stopped pod bills for disk only and is removed
   by whoever launched it or a later sweep.

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
| Pod preempted or stopped | SIGTERM path uploads what exists |
| Hub unreachable at the end | retried; then exit 70 and outputs are lost |
| Output repo is public | job refused before running |

## Not covered

Scheduling and launching (a launcher script lives with the project that uses this),
multi-node training, and a ROCm image for local machines.
