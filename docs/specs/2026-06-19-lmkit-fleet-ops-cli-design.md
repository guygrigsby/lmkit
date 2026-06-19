# lmkit fleet ops CLI (wp5) — design

**Status:** design (approved direction 2026-06-19). Bead: `lmkit-wp5`.

**Goal:** a single Go binary, `lmkit`, that lets a human see and control all lmkit
training workers across boxes without ssh + systemctl + tail. Make the stack
usable with no agent in the loop, and close the reboot-durability gap.

## Why

The training *engine* (this Python package: pretrain/sft/tokenizer/shard/eval) is
reusable and versioned. The *operational surface* is not: launching a run means
ssh to a box, hand-writing a `systemd-run --user` transient unit, invoking
`run_frontier.sh`, then `journalctl`/`tail metrics.jsonl` to watch it. Two
concrete problems:

1. **No agent-free control.** All of the above is tribal knowledge, not a command.
2. **Transient units don't survive reboot.** Current workers are
   `systemd-run --user` transient units (live in `/run/...`). A box reboot wipes
   them — training does *not* auto-resume, despite linger being enabled. That
   violates durable-by-default.

`lmkit run` creating *persistent, enabled* units fixes (2); the read/control
commands fix (1).

## Language & placement

Go (matches the new-tools-in-Go rule; clean static binary, mature ssh/http/
systemd story). The Python `lmkit` package stays the training library. The Go
binary becomes the canonical `lmkit` on PATH; training subcommands not owned by
the CLI delegate to the venv Python (`python -m lmkit.<x>`).

Polyglot repo layout:

```
lmkit/                  # existing Python package (unchanged)
cmd/lmkit/main.go       # Go entry point
internal/fleet/         # fleet.toml + lmkit.toml manifest parsing
internal/unit/          # systemd unit rendering
internal/exporter/      # :9837 metrics client (parse lmkit_*)
internal/remote/        # ssh exec wrapper (systemctl/journalctl)
go.mod                  # at repo root
```

## Ubiquitous language

- **box** — a training host (trig, bee, a cloud instance). Reached by ssh + an
  exporter HTTP endpoint.
- **worker** — one training run, identified `project/run` (e.g. `moe/16e`).
  Backed by a persistent systemd --user unit `lmkit-<project>-<run>.service`.
- **fleet** — the set of boxes, declared in `~/.config/lmkit/fleet.toml`.
- **manifest** — per-project `lmkit.toml` declaring how each worker launches.

## Anti-corruption boundaries

The CLI never reaches into vendor internals directly; each external system is
wrapped in one `internal/` package so the rest of the code speaks lmkit terms:

- **systemd** — `internal/remote` + `internal/unit`. The CLI speaks `start /
  stop / run(worker)`, not raw `systemctl` strings.
- **Prometheus exporter** — `internal/exporter` turns `:9837` text into a
  `WorkerStatus` struct. The metric names (`lmkit_step`, ...) live only here.
- **ssh** — `internal/remote` owns transport; commands compose `RemoteCmd`,
  not shell strings.

## Fleet config — `~/.config/lmkit/fleet.toml`

```toml
[box.trig]
ssh       = "trig"                    # ssh alias / host
exporter  = "http://100.99.25.17:9837"
[box.bee]
ssh       = "bee"
exporter  = "http://100.82.113.122:9837"
```

## Manifest — per-project `lmkit.toml`

Declares each worker so launch is declarative, replacing `run_frontier.sh`:

```toml
project = "moe"

[[run]]
name    = "16e"
box     = "trig"
venv    = "~/venvs/rocm"
gpu     = "rocm:0"                     # -> HIP_VISIBLE_DEVICES=0
workdir = "~/projects/training/moe"
cmd     = "python train.py --n-experts 16 --out-dir checkpoints-16e --max-steps 200000"

[[run]]
name    = "8e"
box     = "trig"
venv    = "~/venvs/cuda"
gpu     = "cuda:0"                     # -> CUDA_VISIBLE_DEVICES=0
workdir = "~/projects/training/moe"
cmd     = "python train.py --n-experts 8 --out-dir checkpoints-8e --batch-size 2 --grad-accum 32 --max-steps 200000"
```

`gpu = "rocm:0" | "cuda:N"` expands to the right visibility env var. `env`
table (optional) for extras (`AIM_REPO`, `LMKIT_PROM_TEXTFILE_DIR`).

## Persistent unit template (rendered by `lmkit run`)

Written to `~/.config/systemd/user/lmkit-<project>-<run>.service` on the box,
then `daemon-reload && enable --now`. Persistent + enabled + linger (already on)
= survives reboot. `Restart=on-failure` = auto-recovers crashes (durable rule).

```ini
[Unit]
Description=lmkit {project}/{run}
After=default.target

[Service]
WorkingDirectory={workdir}
Environment={GPU_ENV}            # HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES
Environment=AIM_REPO=...         # from manifest env, if set
ExecStart={venv}/bin/{cmd}       # cmd[0] resolved against the venv bin
Restart=on-failure
RestartSec=30
TimeoutStopSec=120               # room for SIGTERM checkpoint save

[Install]
WantedBy=default.target
```

SIGTERM on `stop` → pretrain's handler checkpoints and exits cleanly (proven
lossless), so stop/restart is safe.

## Commands (v1)

- `lmkit status [--box X] [--json]` — for each fleet box: GET its exporter,
  parse `lmkit_*` into per-worker rows (project/run, step, progress%, loss,
  tok/s, VRAM, last-seen age, running), and reconcile with `systemctl --user
  is-active` (exporter shows last metrics even if the process died; systemd is
  ground truth for liveness). Prints a table; `--json` for scripts.
- `lmkit logs <project/run> [-f]` — `ssh box journalctl --user -u
  lmkit-<project>-<run> [-f]`.
- `lmkit start <project/run>` / `lmkit stop <project/run>` — `ssh box systemctl
  --user start|stop` the worker unit.
- `lmkit run <project/run>` — read the project manifest, render the persistent
  unit, scp/write it to the box, `daemon-reload && enable --now`. Idempotent:
  re-running updates the unit and restarts. `lmkit run <project>` (no run) =
  all runs in the manifest.
- `lmkit ls` — list defined workers (from manifests) and their unit state.

## Status data model

```go
type WorkerStatus struct {
    Project, Run, Box string
    Step, MaxSteps    int64
    TrainLoss         float64
    TokPerSec         float64
    PeakVRAMGB        float64
    LastSeen          time.Duration  // now - last metric ts
    UnitActive        bool           // systemctl is-active
    Running           bool           // exporter's last-event heuristic
}
```

`Alive` for display = `UnitActive && LastSeen < 2*scrapeInterval`.

## Error handling

- A box unreachable (ssh/http fails) → that box's rows show `unreachable`, other
  boxes still render. Never abort the whole `status` on one box (be-a-good-user
  / no-batch-abort ethos).
- `run` refuses if the manifest entry is malformed or the venv/workdir doesn't
  exist on the box (pre-flight ssh check), with a clear message.

## Testing (TDD)

Pure functions get tests; ssh/systemctl shells stay thin:
- `internal/fleet`: parse fleet.toml + lmkit.toml (valid, missing fields, bad
  gpu spec).
- `internal/unit`: render a manifest run → expected unit file (golden test);
  gpu spec → env var mapping.
- `internal/exporter`: parse a captured `:9837` payload → `[]WorkerStatus`
  (incl. terminal/`running=0`, missing fields).

## Out of scope (v1)

- No per-box agent/daemon (ssh + the existing exporter suffice).
- No web UI (Grafana owns dashboards).
- NAT'd cloud boxes with no inbound exporter reachability — ssh-reachable boxes
  work; note the gap, revisit if cloud workers need it (the prom Pushgateway
  path, bead `lmkit-vtd`, would also help here).

## Build order

1. `internal/fleet` — fleet.toml + manifest parse (+ tests).
2. `internal/exporter` + `lmkit status` (+ tests).
3. `internal/remote` ssh wrapper + `lmkit start/stop/logs`.
4. `internal/unit` render + `lmkit run` (persistent deploy) (+ golden tests).
5. Migrate the live `moe` arms to a `lmkit.toml` and convert their transient
   units to persistent via `lmkit run` — closes the reboot-durability hole and
   retires `run_frontier.sh`.
