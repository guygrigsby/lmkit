package metrics

import (
	"fmt"
	"strings"
	"testing"

	"github.com/guygrigsby/lmkit/internal/fleet"
	"github.com/guygrigsby/lmkit/internal/remote"
)

// scriptRunner is a fake remote.Runner that returns a canned (stdout, err) for
// each command, matched by a substring of the joined argv. It records calls.
type scriptRunner struct {
	// responses maps an argv-substring to the canned result.
	responses []scriptResp
	calls     []remote.Cmd
}

type scriptResp struct {
	match  string
	stdout string
	err    error
}

func (s *scriptRunner) Run(c remote.Cmd) (string, error) {
	s.calls = append(s.calls, c)
	joined := strings.Join(c.Args, " ")
	for _, r := range s.responses {
		if strings.Contains(joined, r.match) {
			return r.stdout, r.err
		}
	}
	return "", fmt.Errorf("scriptRunner: no canned response for %q", joined)
}

func sampleRun() fleet.Run {
	return fleet.Run{
		Name:    "16e",
		Box:     "trig",
		Venv:    "~/venvs/rocm",
		GPU:     "rocm:0",
		Workdir: "~/projects/training/moe",
		OutDir:  "checkpoints-16e",
		Cmd:     "python train.py",
	}
}

const goodTail = `{"event":"start","step":0,"params":124000000,"max_steps":200000,"ts":900.0}
{"event":"train","step":1500,"train_loss":2.4,"tok_per_sec":42000.0,"tokens_seen":3072000,"peak_vram_gb":18.5,"ts":1000.0}`

func TestAssembleActiveGoodTail(t *testing.T) {
	r := &scriptRunner{responses: []scriptResp{
		{match: "is-active", stdout: "active\n"},
		{match: "tail", stdout: goodTail},
	}}
	// now = 1060 -> last metric ts 1000 -> LastSeen 60s
	ws := Assemble("moe", sampleRun(), "trig", r, 1060.0)
	if !ws.Reachable {
		t.Errorf("Reachable = false, want true")
	}
	if !ws.UnitActive {
		t.Errorf("UnitActive = false, want true")
	}
	if !ws.Running {
		t.Errorf("Running = false, want true")
	}
	if ws.Step != 1500 || ws.MaxSteps != 200000 {
		t.Errorf("Step/MaxSteps = %d/%d, want 1500/200000", ws.Step, ws.MaxSteps)
	}
	if ws.TrainLoss != 2.4 {
		t.Errorf("TrainLoss = %v, want 2.4", ws.TrainLoss)
	}
	if ws.TokPerSec != 42000.0 {
		t.Errorf("TokPerSec = %v, want 42000", ws.TokPerSec)
	}
	if ws.PeakVRAMGB != 18.5 {
		t.Errorf("PeakVRAMGB = %v, want 18.5", ws.PeakVRAMGB)
	}
	if ws.LastSeen.Seconds() != 60 {
		t.Errorf("LastSeen = %v, want 60s", ws.LastSeen)
	}
	if ws.Project != "moe" || ws.Run != "16e" || ws.Box != "trig" {
		t.Errorf("identity wrong: %+v", ws)
	}
}

func TestAssembleSSHTransportFailureUnreachable(t *testing.T) {
	// is-active itself fails as a transport error (ssh dial failed). The shape
	// we use: stderr mentions connection refused; importantly stdout is empty,
	// which is how we tell transport failure from an inactive-unit non-zero exit.
	r := &scriptRunner{responses: []scriptResp{
		{match: "is-active", stdout: "", err: fmt.Errorf("ssh trig: exit status 255: ssh: connect to host trig port 22: Connection refused")},
		{match: "tail", stdout: "", err: fmt.Errorf("ssh trig: exit status 255: Connection refused")},
	}}
	ws := Assemble("moe", sampleRun(), "trig", r, 1060.0)
	if ws.Reachable {
		t.Errorf("Reachable = true, want false on ssh transport failure")
	}
}

func TestAssembleInactiveUnitStillReachable(t *testing.T) {
	// is-active exits non-zero but prints "inactive" on stdout: the unit is
	// stopped, the box is fine. Tail file is missing (non-zero, empty stdout
	// but a 'No such file' style stderr) -> reachable, just no metrics.
	r := &scriptRunner{responses: []scriptResp{
		{match: "is-active", stdout: "inactive\n", err: fmt.Errorf("ssh trig: exit status 3")},
		{match: "tail", stdout: "", err: fmt.Errorf("ssh trig: exit status 1: tail: cannot open ... No such file or directory")},
	}}
	ws := Assemble("moe", sampleRun(), "trig", r, 1060.0)
	if !ws.Reachable {
		t.Errorf("Reachable = false, want true (box answered, unit just inactive)")
	}
	if ws.UnitActive {
		t.Errorf("UnitActive = true, want false")
	}
	if ws.Running {
		t.Errorf("Running = true, want false (no metrics)")
	}
}

func TestAssembleUnitNameAndTailCommand(t *testing.T) {
	r := &scriptRunner{responses: []scriptResp{
		{match: "is-active", stdout: "active\n"},
		{match: "tail", stdout: goodTail},
	}}
	Assemble("moe", sampleRun(), "trig", r, 1060.0)
	if len(r.calls) != 2 {
		t.Fatalf("got %d calls, want 2", len(r.calls))
	}
	isActive := strings.Join(r.calls[0].Args, " ")
	if !strings.Contains(isActive, "lmkit-moe-16e") {
		t.Errorf("is-active argv missing unit name: %q", isActive)
	}
	tail := strings.Join(r.calls[1].Args, " ")
	if !strings.Contains(tail, "checkpoints-16e/metrics.jsonl") {
		t.Errorf("tail argv missing metrics path: %q", tail)
	}
	for _, c := range r.calls {
		if c.Host != "trig" {
			t.Errorf("call host = %q, want trig", c.Host)
		}
	}
}
