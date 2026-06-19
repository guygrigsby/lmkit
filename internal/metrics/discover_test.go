package metrics

import (
	"fmt"
	"strings"
	"testing"
)

func TestRunFromMetricsPath(t *testing.T) {
	tests := []struct {
		path        string
		wantProject string
		wantRun     string
		wantOK      bool
	}{
		{"~/projects/training/moe/checkpoints-16e/metrics.jsonl", "moe", "16e", true},
		{"~/projects/training/lm-100m-en/checkpoints-sft/metrics.jsonl", "lm-100m-en", "sft", true},
		{"~/projects/training/anneal/checkpoints/metrics.jsonl", "anneal", "default", true},
		// absolute runs-root also works
		{"/home/x/projects/training/moe/checkpoints-8e/metrics.jsonl", "moe", "8e", true},
		// junk: too short
		{"metrics.jsonl", "", "", false},
		{"checkpoints-16e/metrics.jsonl", "", "", false},
		{"", "", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.path, func(t *testing.T) {
			project, run, ok := runFromMetricsPath(tt.path)
			if ok != tt.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tt.wantOK)
			}
			if !ok {
				return
			}
			if project != tt.wantProject || run != tt.wantRun {
				t.Errorf("got %q/%q, want %q/%q", project, run, tt.wantProject, tt.wantRun)
			}
		})
	}
}

func TestDiscoverBoxTwoWorkers(t *testing.T) {
	lsOut := "~/projects/training/moe/checkpoints-16e/metrics.jsonl\n" +
		"~/projects/training/lm-100m-en/checkpoints-sft/metrics.jsonl\n"
	r := &scriptRunner{responses: []scriptResp{
		{match: "ls ", stdout: lsOut},
		{match: "is-active", stdout: "active\n"},
		{match: "tail", stdout: goodTail},
	}}
	got := DiscoverBox(r, "trig", "trig", "~/projects/training", 1060.0)
	if len(got) != 2 {
		t.Fatalf("got %d workers, want 2: %+v", len(got), got)
	}
	byKey := map[string]WorkerStatus{}
	for _, ws := range got {
		byKey[ws.Project+"/"+ws.Run] = ws
	}
	moe, ok := byKey["moe/16e"]
	if !ok {
		t.Fatalf("missing moe/16e: %+v", got)
	}
	if !moe.Reachable || !moe.UnitActive || !moe.Running {
		t.Errorf("moe/16e flags wrong: %+v", moe)
	}
	if moe.Box != "trig" {
		t.Errorf("moe/16e box = %q, want trig", moe.Box)
	}
	if moe.Step != 1500 || moe.MaxSteps != 200000 {
		t.Errorf("moe/16e step/max = %d/%d, want 1500/200000", moe.Step, moe.MaxSteps)
	}
	if moe.LastSeen.Seconds() != 60 {
		t.Errorf("moe/16e LastSeen = %v, want 60s", moe.LastSeen)
	}
	if _, ok := byKey["lm-100m-en/sft"]; !ok {
		t.Fatalf("missing lm-100m-en/sft: %+v", got)
	}
}

func TestDiscoverBoxTransportFailure(t *testing.T) {
	// ssh dial failed: ssh exits 255. One Reachable=false row, no workers.
	r := &scriptRunner{responses: []scriptResp{
		{match: "ls ", stdout: "", err: fmt.Errorf("ssh trig: exit status 255: ssh: connect to host trig port 22: Connection refused")},
	}}
	got := DiscoverBox(r, "trig", "trig", "~/projects/training", 1060.0)
	if len(got) != 1 {
		t.Fatalf("got %d rows, want 1: %+v", len(got), got)
	}
	if got[0].Reachable {
		t.Errorf("Reachable = true, want false on transport failure")
	}
	if got[0].Box != "trig" {
		t.Errorf("Box = %q, want trig", got[0].Box)
	}
}

func TestDiscoverBoxNoMatch(t *testing.T) {
	// ls found nothing: non-zero exit (not 255) with a normal stderr. Box is
	// reachable, just has no workers -> empty slice.
	r := &scriptRunner{responses: []scriptResp{
		{match: "ls ", stdout: "", err: fmt.Errorf("ssh trig: exit status 2: ls: cannot access '~/projects/training/*/checkpoints*/metrics.jsonl': No such file or directory")},
	}}
	got := DiscoverBox(r, "trig", "trig", "~/projects/training", 1060.0)
	if len(got) != 0 {
		t.Fatalf("got %d rows, want 0 (reachable, no workers): %+v", len(got), got)
	}
}

func TestDiscoverBoxLsCommand(t *testing.T) {
	r := &scriptRunner{responses: []scriptResp{
		{match: "ls ", stdout: ""},
	}}
	DiscoverBox(r, "trig", "trig", "~/projects/training", 1060.0)
	if len(r.calls) == 0 {
		t.Fatal("no calls recorded")
	}
	ls := strings.Join(r.calls[0].Args, " ")
	if !strings.Contains(ls, "~/projects/training/*/checkpoints*/metrics.jsonl") {
		t.Errorf("ls argv missing glob: %q", ls)
	}
}
