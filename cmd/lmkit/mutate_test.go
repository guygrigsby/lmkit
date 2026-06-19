package main

import (
	"errors"
	"strings"
	"testing"

	"github.com/guygrigsby/lmkit/internal/fleet"
	"github.com/guygrigsby/lmkit/internal/remote"
)

// scriptRunner is a Runner test double: it records every Cmd it receives and,
// for each, returns the first scripted reply whose key is a substring of the
// joined argv. An unmatched command returns ("", nil) (a benign success), so a
// test only scripts the commands whose outcome it cares about.
type scriptRunner struct {
	replies []reply
	calls   []string // joined argv, in order
}

type reply struct {
	match  string
	stdout string
	err    error
}

func (s *scriptRunner) Run(c remote.Cmd) (string, error) {
	joined := strings.Join(c.Args, " ")
	s.calls = append(s.calls, joined)
	for _, r := range s.replies {
		if strings.Contains(joined, r.match) {
			return r.stdout, r.err
		}
	}
	return "", nil
}

func (s *scriptRunner) script(match, stdout string, err error) {
	s.replies = append(s.replies, reply{match, stdout, err})
}

// order asserts the recorded calls contain each of subs as a substring, in the
// given relative order (other calls may interleave).
func (s *scriptRunner) assertOrder(t *testing.T, subs ...string) {
	t.Helper()
	idx := 0
	for _, call := range s.calls {
		if idx < len(subs) && strings.Contains(call, subs[idx]) {
			idx++
		}
	}
	if idx != len(subs) {
		t.Fatalf("expected calls in order %v\ngot calls:\n  %s", subs, strings.Join(s.calls, "\n  "))
	}
}

func (s *scriptRunner) sawAny(sub string) bool {
	for _, call := range s.calls {
		if strings.Contains(call, sub) {
			return true
		}
	}
	return false
}

func testRun() fleet.Run {
	return fleet.Run{
		Name: "16e", Box: "trig", Venv: "~/venvs/rocm", GPU: "rocm:0",
		Workdir: "~/projects/training/moe", OutDir: "checkpoints-16e",
		Cmd: "python train.py --n-experts 16",
	}
}

// run happy path: lock, snapshot, write+mv unit, daemon-reload, enable --now,
// verify is-active=active, release lock. No rollback.
func TestRunHappyPath(t *testing.T) {
	r := &scriptRunner{}
	r.script("is-active", "active\n", nil) // verification passes
	if err := doRun(r, "moe", testRun(), "trig", ""); err != nil {
		t.Fatalf("doRun: %v", err)
	}
	r.assertOrder(t,
		"mkdir",
		"cat",
		"mv",
		"daemon-reload",
		"enable --now",
		"is-active",
		"rmdir",
	)
	if r.sawAny("disable") {
		t.Fatalf("happy path should not disable:\n%s", strings.Join(r.calls, "\n"))
	}
}

// run: enable --now fails -> full rollback. disable --now, restore unit, a
// second daemon-reload, then release the lock.
func TestRunEnableFailsRollback(t *testing.T) {
	r := &scriptRunner{}
	r.script("enable --now", "", errors.New("enable boom"))
	err := doRun(r, "moe", testRun(), "trig", "")
	if err == nil {
		t.Fatalf("expected error when enable fails")
	}
	r.assertOrder(t,
		"mkdir", // lock
		"mv",    // write unit
		"daemon-reload",
		"enable --now",  // fails here
		"disable",       // undo enable
		"daemon-reload", // undo daemon-reload (after file restore)
		"rmdir",         // release lock
	)
}

// run: enable succeeds but verify reports inactive -> fail closed -> full
// rollback (disable, restore unit, reload, unlock).
func TestRunVerifyInactiveRollback(t *testing.T) {
	r := &scriptRunner{}
	r.script("is-active", "inactive\n", errors.New("inactive"))
	err := doRun(r, "moe", testRun(), "trig", "")
	if err == nil {
		t.Fatalf("expected error when verify shows inactive")
	}
	r.assertOrder(t, "enable --now", "is-active", "disable", "rmdir")
}

// run: lock contention -> the very first mkdir fails -> abort immediately,
// nothing else touched, and NO rmdir (we never acquired the lock).
func TestRunLockContentionAborts(t *testing.T) {
	r := &scriptRunner{}
	r.script("mkdir", "", errors.New("File exists"))
	err := doRun(r, "moe", testRun(), "trig", "")
	if err == nil {
		t.Fatalf("expected error on lock contention")
	}
	if r.sawAny("mv") || r.sawAny("enable") || r.sawAny("daemon-reload") {
		t.Fatalf("nothing should run after a failed lock:\n%s", strings.Join(r.calls, "\n"))
	}
	if r.sawAny("rmdir") {
		t.Fatalf("must not release a lock it never acquired:\n%s", strings.Join(r.calls, "\n"))
	}
}

// start happy path: lock, start, verify active, unlock.
func TestStartHappyPath(t *testing.T) {
	r := &scriptRunner{}
	r.script("is-active", "active\n", nil)
	if err := doStart(r, "moe", "16e", "trig"); err != nil {
		t.Fatalf("doStart: %v", err)
	}
	r.assertOrder(t, "mkdir", "start", "is-active", "rmdir")
	if r.sawAny("stop") {
		t.Fatalf("happy start should not stop")
	}
}

// start: verify inactive -> rollback issues stop, then unlock.
func TestStartVerifyInactiveRollsBackWithStop(t *testing.T) {
	r := &scriptRunner{}
	r.script("is-active", "inactive\n", errors.New("inactive"))
	err := doStart(r, "moe", "16e", "trig")
	if err == nil {
		t.Fatalf("expected error when start does not activate")
	}
	r.assertOrder(t, "start", "is-active", "stop", "rmdir")
}

// stop happy path: lock, stop, verify NOT active, unlock.
func TestStopHappyPath(t *testing.T) {
	r := &scriptRunner{}
	// is-active after stop returns inactive (non-zero) -> not active -> good.
	r.script("is-active", "inactive\n", errors.New("inactive"))
	if err := doStop(r, "moe", "16e", "trig"); err != nil {
		t.Fatalf("doStop: %v", err)
	}
	r.assertOrder(t, "mkdir", "stop", "is-active", "rmdir")
	if r.sawAny("kill") || r.sawAny("SIGKILL") {
		t.Fatalf("stop must never SIGKILL:\n%s", strings.Join(r.calls, "\n"))
	}
}

// stop: unit stays active after stop -> error, no SIGKILL.
func TestStopStaysActiveErrors(t *testing.T) {
	r := &scriptRunner{}
	r.script("is-active", "active\n", nil) // still active
	err := doStop(r, "moe", "16e", "trig")
	if err == nil {
		t.Fatalf("expected error when unit stays active after stop")
	}
	if r.sawAny("kill") || r.sawAny("SIGKILL") || r.sawAny("-9") {
		t.Fatalf("stop must never escalate to SIGKILL:\n%s", strings.Join(r.calls, "\n"))
	}
}
