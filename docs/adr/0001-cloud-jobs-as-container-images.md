# 1. Cloud GPU jobs run as container images with outputs in a private hub repo

Status: Accepted

## Context

Cloud GPU runs were done by renting a pod, installing dependencies over ssh, copying scripts
by hand and watching the session. That is not reproducible, needs a connected operator, and
drifts from the repository. Training runs here take hours to days, and the connection to the
cheapest datacenter dropped repeatedly during benchmarking.

## Decision

- A job is a container image built in CI from this repository and pushed to GHCR
  (`ghcr.io/guygrigsby/lmkit-cuda`). Dependencies are pinned in the image.
- The container entrypoint (`lmkit.cloudjob`) runs one job configured by environment
  variables, uploads its outputs to a private HuggingFace repo on the `guygrigsby` account,
  then exits. Nobody stays connected.
- Outputs go to HuggingFace, not a cloud network volume, so a job can run in any datacenter.
- Model weights are not baked into the image; the job downloads them at start.

Design: `docs/specs/2026-11-19-cloud-job-image-design.md`.

## Consequences

- Any change to the training code requires a CI image build before a cloud run, which is the
  intended source of truth but adds minutes to the loop.
- A hub write token is present in the pod environment while a job runs. A token scoped to the
  one results repo limits the exposure.
- Results survive the pod. Checkpoints for resumable runs travel through the hub, so large
  checkpoints cost upload time on every save.
- A ROCm variant for the local trig machine is not covered and would be a separate image.

## Alternatives considered

- Network volume for data and outputs: fast and persistent, but pinned to one datacenter,
  which would have limited GPU availability.
- Keep installing at runtime over ssh: rejected as above.
- Docker Hub instead of GHCR: needs a separate credential on the build machine; GHCR uses
  the Actions token.
