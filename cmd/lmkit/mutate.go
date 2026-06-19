package main

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/guygrigsby/lmkit/internal/fleet"
	"github.com/guygrigsby/lmkit/internal/metrics"
	"github.com/guygrigsby/lmkit/internal/remote"
	"github.com/guygrigsby/lmkit/internal/txn"
	"github.com/guygrigsby/lmkit/internal/unit"
)

// Mutating commands (run/start/stop) are ACID transactions over ssh:
//
//   - Isolated by a per-worker advisory lock. The lock is a `mkdir` mutex: the
//     directory ~/.config/lmkit/locks/<p>-<r> is created atomically (mkdir
//     fails if it already exists, so a held lock fails closed). It is the FIRST
//     txn step, so any rollback releases it; release is also the LAST step so a
//     successful txn releases it too. (mkdir is atomic over ssh and needs no
//     background holder, unlike flock; chosen for that simplicity.)
//   - Atomic + consistent via internal/txn: any failed step rolls back all
//     completed steps in reverse, leaving the box exactly as it was.
//   - Fail closed: each command ends by verifying the intended end state
//     (is-active after run/start, ! is-active after stop). An unmet or
//     indeterminate state is treated as failure -> rollback.

// runRun handles `lmkit run <project/run>` (one worker) and `lmkit run
// <project>` (every run in the manifest, each its own independent transaction;
// one failing rolls back only itself and is reported, the rest proceed).
func runRun(args []string) error {
	fs := newFlagSet("run")
	manifestPath := fs.String("manifest", "lmkit.toml", "path to the project manifest")
	pos, err := parseArgs(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return fmt.Errorf("usage: lmkit run <project/run | project> [--manifest PATH]")
	}
	arg := pos[0]
	runner := remote.NewSSHRunner()

	// project/run form -> single worker.
	if strings.Contains(arg, "/") {
		project, run, err := parseWorker(arg)
		if err != nil {
			return err
		}
		w, err := resolveWorker(*manifestPath, project, run)
		if err != nil {
			return err
		}
		return doRun(runner, w.project, w.run, w.sshHost, w.gpuWrap)
	}

	// bare project -> all runs, each an independent transaction.
	man, err := fleet.LoadManifest(*manifestPath)
	if err != nil {
		return err
	}
	if man.Project != arg {
		return fmt.Errorf("manifest project is %q, not %q", man.Project, arg)
	}
	flt, err := fleet.LoadFleet(fleetConfigPath())
	if err != nil {
		return err
	}
	var failed int
	for _, rn := range man.Run {
		box, ok := flt.Box[rn.Box]
		if !ok {
			fmt.Fprintf(os.Stderr, "lmkit run %s/%s: box %q not in fleet config\n", man.Project, rn.Name, rn.Box)
			failed++
			continue
		}
		if err := doRun(runner, man.Project, rn, box.SSH, box.GpuWrap); err != nil {
			fmt.Fprintf(os.Stderr, "lmkit run %s/%s: %v\n", man.Project, rn.Name, err)
			failed++
			continue
		}
		fmt.Printf("lmkit run %s/%s: ok\n", man.Project, rn.Name)
	}
	if failed > 0 {
		return fmt.Errorf("%d of %d runs failed", failed, len(man.Run))
	}
	return nil
}

// runStartStop handles the `start` and `stop` subcommands, which share their
// argument shape (a single project/run). op is "start" or "stop".
func runStartStop(op string, args []string) error {
	fs := newFlagSet(op)
	manifestPath := fs.String("manifest", "lmkit.toml", "path to the project manifest")
	all := fs.Bool("all", false, "apply to every worker discovered across the fleet")
	boxFilter := fs.String("box", "", "with --all, limit to this box")
	pos, err := parseArgs(fs, args)
	if err != nil {
		return err
	}
	runner := remote.NewSSHRunner()

	if *all {
		if len(pos) != 0 {
			return fmt.Errorf("usage: lmkit %s --all [--box NAME]", op)
		}
		return startStopAll(op, runner, *boxFilter)
	}

	if len(pos) != 1 {
		return fmt.Errorf("usage: lmkit %s <project/run> [--manifest PATH] | --all [--box NAME]", op)
	}
	project, run, err := parseWorker(pos[0])
	if err != nil {
		return err
	}
	w, err := resolveWorker(*manifestPath, project, run)
	if err != nil {
		return err
	}
	if op == "start" {
		return doStart(runner, w.project, w.run.Name, w.sshHost)
	}
	return doStop(runner, w.project, w.run.Name, w.sshHost)
}

// startStopAll applies op (start|stop) to every worker discovered across the
// fleet (no manifest) — each its own ACID transaction; one failure never aborts
// the rest. Discovery is the same fleet glob `status` uses, so an unreachable
// box is reported and skipped. (A worker deployed but never run has no
// metrics.jsonl and so is not discovered — start it by name once.)
func startStopAll(op string, runner remote.Runner, boxFilter string) error {
	flt, err := fleet.LoadFleet(fleetConfigPath())
	if err != nil {
		return err
	}
	workers := gatherStatuses(flt, runner, float64(time.Now().Unix()), boxFilter)
	var acted, failed int
	for _, w := range workers {
		if !w.Reachable {
			fmt.Fprintf(os.Stderr, "lmkit %s: box %q unreachable\n", op, w.Box)
			failed++
			continue
		}
		box, ok := flt.Box[w.Box]
		if !ok {
			continue
		}
		acted++
		var e error
		if op == "start" {
			e = doStart(runner, w.Project, w.Run, box.SSH)
		} else {
			e = doStop(runner, w.Project, w.Run, box.SSH)
		}
		if e != nil {
			fmt.Fprintf(os.Stderr, "lmkit %s %s/%s: %v\n", op, w.Project, w.Run, e)
			failed++
			continue
		}
		fmt.Printf("lmkit %s %s/%s: ok\n", op, w.Project, w.Run)
	}
	if acted == 0 && failed == 0 {
		return fmt.Errorf("no workers discovered across the fleet")
	}
	if failed > 0 {
		return fmt.Errorf("%d %s op(s) failed", failed, op)
	}
	return nil
}

// lockDir is the per-worker lock directory path (a leading ~/ is expanded by
// the remote login shell ssh runs the argv through).
func lockDir(project, run string) string {
	return fmt.Sprintf("~/.config/lmkit/locks/%s-%s", project, run)
}

// unitPath is the persistent unit file path on the box.
func unitPath(project, run string) string {
	return fmt.Sprintf("~/.config/systemd/user/%s.service", metrics.UnitName(project, run))
}

// runCmd issues one remote command through the runner, discarding stdout, and
// wraps any error with a label.
func runCmd(r remote.Runner, host, label string, args ...string) error {
	if _, err := r.Run(remote.Cmd{Host: host, Args: args}); err != nil {
		return fmt.Errorf("%s: %w", label, err)
	}
	return nil
}

// isActive reports whether `systemctl --user is-active <unit>` says active. A
// non-zero exit (inactive/failed/...) is not an error here: it is the answer.
func isActive(r remote.Runner, host, unit string) bool {
	out, _ := r.Run(remote.Cmd{Host: host, Args: []string{"systemctl", "--user", "is-active", unit}})
	return strings.TrimSpace(out) == "active"
}

// lockStep builds the lock acquire/release pair. acquire (Do) is the mkdir
// mutex; release (Undo) the rmdir, run on rollback. The caller must also append
// releaseStep so a successful txn releases the lock.
func lockStep(r remote.Runner, host, project, run string) txn.Step {
	dir := lockDir(project, run)
	return txn.Step{
		Name: "acquire lock",
		Do: func() error {
			// mkdir -p the parent (idempotent), then mkdir the lock dir
			// itself (atomic: fails if held).
			return runCmd(r, host, "acquire lock",
				"mkdir", "-p", "~/.config/lmkit/locks", "&&", "mkdir", dir)
		},
		Undo: func() error {
			return runCmd(r, host, "release lock", "rmdir", dir)
		},
	}
}

// releaseStep releases the lock on the success path (its Do). It has no Undo:
// once released there is nothing to compensate.
func releaseStep(r remote.Runner, host, project, run string) txn.Step {
	dir := lockDir(project, run)
	return txn.Step{
		Name: "release lock",
		Do:   func() error { return runCmd(r, host, "release lock", "rmdir", dir) },
	}
}

// doRun deploys (or updates) a worker's persistent unit and starts it, as one
// ACID transaction. Idempotent: re-running rewrites the unit and restarts.
func doRun(r remote.Runner, project string, run fleet.Run, host, gpuWrap string) error {
	unitText, err := unit.Render(project, run, gpuWrap)
	if err != nil {
		return err
	}
	name := metrics.UnitName(project, run.Name)
	path := unitPath(project, run.Name)
	tmp := path + ".tmp"

	var prev string // snapshot of the existing unit file (empty == absent)

	var tx txn.Txn
	tx.Add(lockStep(r, host, project, run.Name))

	tx.Add(txn.Step{
		Name: "snapshot unit",
		Do: func() error {
			// `cat` of an absent file errors; treat that as "no prior unit".
			out, _ := r.Run(remote.Cmd{Host: host, Args: []string{"cat", path}})
			prev = out
			return nil
		},
		// no Undo: snapshotting changes nothing.
	})

	tx.Add(txn.Step{
		Name: "write unit",
		Do: func() error {
			// Write to a temp path then atomically mv into place, so a partial
			// transfer never yields a corrupt unit. The unit text is delivered
			// over stdin to `cat >`; the remote login shell ssh joins the argv
			// through handles the redirection and ~ expansion.
			if _, err := r.Run(remote.Cmd{
				Host:  host,
				Args:  []string{"mkdir", "-p", "~/.config/systemd/user", "&&", "cat", ">", tmp, "&&", "mv", tmp, path},
				Stdin: unitText,
			}); err != nil {
				return fmt.Errorf("write unit: %w", err)
			}
			return nil
		},
		Undo: func() error {
			if prev == "" {
				return runCmd(r, host, "remove unit", "rm", "-f", path)
			}
			_, err := r.Run(remote.Cmd{
				Host:  host,
				Args:  []string{"cat", ">", tmp, "&&", "mv", tmp, path},
				Stdin: prev,
			})
			return err
		},
	})

	tx.Add(txn.Step{
		Name: "daemon-reload",
		Do:   func() error { return runCmd(r, host, "daemon-reload", "systemctl", "--user", "daemon-reload") },
		// Undo runs AFTER the unit file is restored (steps roll back in
		// reverse), so re-reloading picks up the restored/removed unit.
		Undo: func() error { return runCmd(r, host, "daemon-reload", "systemctl", "--user", "daemon-reload") },
	})

	// `enable --now` is non-atomic (it enables the symlink AND starts the
	// unit), so a partial failure can leave either applied. Arm the disable
	// compensation BEFORE the action so it runs even if enable's own Do fails:
	// a no-op guard step whose Undo is `disable --now`, followed by the enable.
	tx.Add(txn.Step{
		Name: "arm enable cleanup",
		Do:   func() error { return nil },
		Undo: func() error { return runCmd(r, host, "disable --now", "systemctl", "--user", "disable", "--now", name) },
	})
	tx.Add(txn.Step{
		Name: "enable --now",
		Do:   func() error { return runCmd(r, host, "enable --now", "systemctl", "--user", "enable", "--now", name) },
	})

	tx.Add(txn.Step{
		Name: "verify is-active",
		Do: func() error {
			if !isActive(r, host, name) {
				return fmt.Errorf("verify: unit %s is not active after enable --now", name)
			}
			return nil
		},
		// no Undo: verification changes nothing.
	})

	tx.Add(releaseStep(r, host, project, run.Name))
	return tx.Run()
}

// doStart starts an already-deployed worker, verifying it reaches active.
func doStart(r remote.Runner, project, run, host string) error {
	name := metrics.UnitName(project, run)
	var tx txn.Txn
	tx.Add(lockStep(r, host, project, run))
	tx.Add(txn.Step{
		Name: "start",
		Do:   func() error { return runCmd(r, host, "start", "systemctl", "--user", "start", name) },
		Undo: func() error { return runCmd(r, host, "stop", "systemctl", "--user", "stop", name) },
	})
	tx.Add(txn.Step{
		Name: "verify is-active",
		Do: func() error {
			if !isActive(r, host, name) {
				return fmt.Errorf("verify: unit %s did not reach active after start", name)
			}
			return nil
		},
	})
	tx.Add(releaseStep(r, host, project, run))
	return tx.Run()
}

// doStop stops a worker with SIGTERM (the unit's TimeoutStopSec gives the
// engine room to checkpoint), then verifies it is no longer active. A unit that
// stays active is reported as a failure; it is NEVER escalated to SIGKILL.
func doStop(r remote.Runner, project, run, host string) error {
	name := metrics.UnitName(project, run)
	var tx txn.Txn
	tx.Add(lockStep(r, host, project, run))
	tx.Add(txn.Step{
		Name: "stop",
		Do:   func() error { return runCmd(r, host, "stop", "systemctl", "--user", "stop", name) },
		// no Undo: re-starting a unit the user asked to stop is the wrong
		// "safe" state; a failed stop is surfaced, not silently reverted.
	})
	tx.Add(txn.Step{
		Name: "verify ! is-active",
		Do: func() error {
			if isActive(r, host, name) {
				return fmt.Errorf("verify: unit %s still active after stop (not escalating to SIGKILL)", name)
			}
			return nil
		},
	})
	tx.Add(releaseStep(r, host, project, run))
	return tx.Run()
}
