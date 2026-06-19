# lmkit fleet ops CLI (wp5) — design

**Status:** design (approved 2026-06-19). Bead: `lmkit-wp5`.

**Goal:** a single Go binary, `lmkit`, run locally, that lets a human see and
control all lmkit training workers across boxes without ssh + systemctl + tail.
Make the stack usable with no agent in the loop, and close the
reboot-durability gap.

## Why

The training *engine* (this Python package: pretrain/sft/tokenizer/shard/eval)
is reusable and versioned. The *operational surface* is not: launching a run
means ssh to a box, hand-writing a `systemd-run --user` transient unit,
invoking `run_frontier.sh`, then `journalctl`/`tail metrics.jsonl` to watch it.
Two concrete problems:

1. **No agent-free control.** All of the above is tribal knowledge, not a command.
2. **Transient units don't survive reboot.** Current workers are
   `systemd-run --user` transient units (live in `/run/...`). A box reboot wipes
   them — training does *not* auto-resume, despite linger being enabled. That
   violates durable-by-default.

`lmkit run` creating *persistent, enabled* units fixes (2); the read/control
commands fix (1).

## Hard constraints (Guy, 2026-06-19)

These shape everything below:

- **Local-only CLI, no remote agent.** The binary runs on the workstation. No
  daemon/sidecar on any box.
- **ssh is the ONLY transport.** No HTTP, no opening exporter ports to the CLI,
  no other channel. Every read and every action is an ssh command to a box.
- **Never reach into the lmkit engine.** The CLI does not import, call, or wrap
  the Python `lmkit` package. A worker's launch command is an opaque string from
  the manifest, executed by systemd. Status is derived from reading the
  artifacts the engine writes (`metrics.jsonl`), never from the engine itself.
- **ACID.** Every mutating command is atomic, fails closed, and reverts to the
  prior state on any error. See "Transactions & failure semantics".

## Language & placement

Go (matches the new-tools-in-Go rule; static binary, mature ssh/exec story).
The Python `lmkit` package remains the training library, entirely separate; the
Go binary is an **ops tool only** and shares nothing with the engine at runtime.

Polyglot repo layout:

```
lmkit/                  # existing Python package (unchanged, untouched by the CLI)
cmd/lmkit/main.go       # Go entry point
internal/fleet/         # fleet.toml + lmkit.toml manifest parsing
internal/unit/          # systemd unit rendering
internal/metrics/       # parse a metrics.jsonl tail (read over ssh) -> status
internal/remote/        # ssh exec wrapper — the sole transport
internal/txn/           # transaction runner: ordered steps + rollback
go.mod                  # at repo root
```

## Ubiquitous language

- **box** — a training host (trig, bee, a cloud instance). Reached by **ssh only**.
- **worker** — one training run, identified `project/run` (e.g. `moe/16e`).
  Backed by a persistent systemd --user unit `lmkit-<project>-<run>.service`.
- **fleet** — the set of boxes, declared in `~/.config/lmkit/fleet.toml`.
- **manifest** — per-project `lmkit.toml` declaring how each worker launches.

## Anti-corruption boundaries

Each external system is wrapped in one `internal/` package so the rest of the
code speaks lmkit terms:

- **ssh** — `internal/remote` owns the only transport. Callers compose a
  `RemoteCmd` (box + argv), never raw shell strings; the package quotes/escapes.
- **systemd** — `internal/unit` + `internal/remote`. The CLI speaks
  `start / stop / run(worker)`, mapped to `systemctl --user` invocations over ssh.
- **engine artifacts** — `internal/metrics` turns a `metrics.jsonl` tail (fetched
  over ssh) into a `WorkerStatus`. The JSONL event schema lives only here. The
  CLI never executes or imports engine code to learn status.

## Fleet config — `~/.config/lmkit/fleet.toml`

```toml
[box.trig]
ssh = "trig"            # ssh alias / host. That is all the CLI needs.
[box.bee]
ssh = "bee"
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
out_dir = "checkpoints-16e"            # metrics.jsonl lives at workdir/out_dir/
cmd     = "python train.py --n-experts 16 --out-dir checkpoints-16e --max-steps 200000"

[[run]]
name    = "8e"
box     = "trig"
venv    = "~/venvs/cuda"
gpu     = "cuda:0"                     # -> CUDA_VISIBLE_DEVICES=0
workdir = "~/projects/training/moe"
out_dir = "checkpoints-8e"
cmd     = "python train.py --n-experts 8 --out-dir checkpoints-8e --batch-size 2 --grad-accum 32 --max-steps 200000"
```

`gpu = "rocm:0" | "cuda:N"` expands to the right visibility env var. `out_dir`
gives the CLI the metrics path without parsing `cmd`. Optional `env` table for
extras (`AIM_REPO`, `LMKIT_PROM_TEXTFILE_DIR`).

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

- `lmkit status [--box X] [--json]` — for each worker (from manifests): one ssh
  to its box runs `systemctl --user is-active lmkit-<p>-<r>` and tails
  `workdir/out_dir/metrics.jsonl`; `internal/metrics` folds the tail into a row
  (step, progress%, loss, tok/s, VRAM, last-seen, alive). Prints a table;
  `--json` for scripts. A box that is unreachable shows `unreachable` and does
  not abort the others (read-only, so no rollback needed — just report).
- `lmkit logs <project/run> [-f]` — `ssh box journalctl --user -u
  lmkit-<project>-<run> [-f]`.
- `lmkit start <project/run>` — mutating; see transactions below.
- `lmkit stop <project/run>` — mutating; SIGTERM via systemctl.
- `lmkit run <project/run>` — mutating; render + deploy the persistent unit.
  `lmkit run <project>` = all runs in the manifest, each its own transaction.
- `lmkit ls` — list defined workers (from manifests) and their unit state.

## Transactions & failure semantics (ACID)

Every mutating command runs through `internal/txn`: an ordered list of steps,
each with a `do` and an `undo`. If any step fails, the runner invokes `undo`
for all completed steps in reverse, leaving the box exactly as it was. The
command then exits non-zero with the failure. Nothing is left half-applied.

- **Atomic.** All-or-nothing per worker. (`run <project>` is N independent
  per-worker transactions; one failing rolls back only itself and is reported.)
- **Consistent.** Each transaction ends in a valid state: either fully applied
  or fully reverted. Unit files are written to a temp path then `mv`'d (atomic
  rename) so a partial transfer never yields a corrupt unit.
- **Isolated.** A transaction takes a per-worker advisory lock on the box
  (`flock` on `~/.config/lmkit/locks/<p>-<r>.lock` over ssh) so concurrent
  `run/start/stop` on the same worker cannot interleave.
- **Durable.** On success the persistent unit + enable symlink are on disk;
  survives reboot.
- **Fail closed.** Every action ends by *verifying* the intended end state over
  ssh (e.g. `is-active` after start, `! is-active` after stop, unit present +
  enabled after run). If verification fails or is indeterminate, the command
  treats it as failure and rolls back — it never reports success on uncertainty.

Per-command transactions:

- **`run`** steps:
  1. snapshot any existing unit file (read its contents, or note absence).
  2. write the new unit to a temp path on the box, `mv` into place. *undo:*
     restore the snapshot (rewrite old contents, or `rm` if none).
  3. `systemctl --user daemon-reload`. *undo:* `daemon-reload` again after the
     file is restored.
  4. `systemctl --user enable --now lmkit-<p>-<r>`. *undo:* `disable --now`.
  5. verify `is-active`. On failure → roll back 4→1.
- **`start`** = `systemctl --user start` then verify `is-active`; failure to
  reach active rolls back with `stop` and reports. (Prior state was stopped, so
  the safe state is stopped.)
- **`stop`** = `systemctl --user stop` (SIGTERM, unit's TimeoutStopSec allows the
  checkpoint save) then verify `! is-active`. If it does not stop within the
  timeout, report failure; do not silently SIGKILL (a stuck stop is surfaced,
  not hidden).

## Status data model

```go
type WorkerStatus struct {
    Project, Run, Box string
    Step, MaxSteps    int64
    TrainLoss         float64
    TokPerSec         float64
    PeakVRAMGB        float64
    LastSeen          time.Duration  // now - last metric ts (from the box clock)
    UnitActive        bool           // systemctl is-active
    Running           bool           // metrics.jsonl last-event heuristic
    Reachable         bool
}
```

`Alive` for display = `Reachable && UnitActive && LastSeen` recent.

## Testing (TDD)

Pure functions get tests; ssh/systemctl shells stay thin:
- `internal/fleet`: parse fleet.toml + lmkit.toml (valid, missing fields, bad
  gpu spec).
- `internal/unit`: render a manifest run → expected unit file (golden test);
  gpu spec → env var mapping.
- `internal/metrics`: parse a captured `metrics.jsonl` tail → `WorkerStatus`
  (incl. terminal/`running=0`, missing fields, mixed event types).
- `internal/txn`: a transaction whose step 3 fails invokes undo for steps 2,1 in
  reverse; assert the recorded undo calls and order. (Inject a fake step runner
  so no real ssh is needed.)

## Out of scope (v1)

- No per-box agent/daemon, no HTTP, no web UI (Grafana owns dashboards; the
  `:9837` exporter stays for Grafana and is unrelated to the CLI).
- NAT'd cloud boxes still need ssh reachability; if a box isn't ssh-reachable
  it's simply `unreachable` in `status` and uncontrollable until it is.

## Build order

1. `internal/remote` (ssh exec) + `internal/fleet` (fleet.toml + manifest parse) (+ tests).
2. `internal/metrics` + `lmkit status` (+ tests).
3. `lmkit logs` (thin ssh journalctl).
4. `internal/txn` + `internal/unit` + `lmkit run` / `start` / `stop` with full
   rollback + verification (+ txn/unit tests).
5. Migrate the live `moe` arms to a `lmkit.toml` and convert their transient
   units to persistent via `lmkit run` — closes the reboot-durability hole and
   retires `run_frontier.sh`.
