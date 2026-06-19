package main

import (
	"strings"
	"testing"
	"time"

	"github.com/guygrigsby/lmkit/internal/metrics"
)

func TestRenderTable(t *testing.T) {
	rows := []metrics.WorkerStatus{
		{
			Project: "moe", Run: "16e", Box: "trig",
			Step: 1500, MaxSteps: 200000,
			TrainLoss: 2.4, TokPerSec: 42000, PeakVRAMGB: 18.5,
			LastSeen:   10 * time.Second,
			UnitActive: true, Running: true, Reachable: true,
		},
		{
			Project: "moe", Run: "8e", Box: "trig",
			Step: 0, MaxSteps: 200000,
			Reachable: false, // unreachable box
		},
		{
			Project: "moe", Run: "32e", Box: "bee",
			Step: 5000, MaxSteps: 100000,
			TrainLoss: 1.8, TokPerSec: 30000, PeakVRAMGB: 22.0,
			LastSeen:   5 * time.Minute, // stale -> not alive
			UnitActive: true, Running: true, Reachable: true,
		},
	}
	out := renderTable(rows)

	// Header columns present.
	for _, col := range []string{"PROJECT/RUN", "BOX", "STEP", "PROGRESS%", "LOSS", "TOK/S", "VRAM", "LAST-SEEN", "ALIVE"} {
		if !strings.Contains(out, col) {
			t.Errorf("table missing column %q\n%s", col, out)
		}
	}
	// Worker identity.
	if !strings.Contains(out, "moe/16e") {
		t.Errorf("missing moe/16e row\n%s", out)
	}
	// Progress for 1500/200000 = 0.75%.
	if !strings.Contains(out, "0.8%") && !strings.Contains(out, "0.75%") {
		t.Errorf("missing progress for 16e\n%s", out)
	}
	// Active + fresh -> alive yes.
	lines := strings.Split(out, "\n")
	var line16e, line8e, line32e string
	for _, l := range lines {
		switch {
		case strings.Contains(l, "moe/16e"):
			line16e = l
		case strings.Contains(l, "moe/8e"):
			line8e = l
		case strings.Contains(l, "moe/32e"):
			line32e = l
		}
	}
	if !strings.Contains(line16e, "yes") {
		t.Errorf("16e should be alive=yes: %q", line16e)
	}
	if !strings.Contains(line8e, "unreachable") {
		t.Errorf("8e should be unreachable: %q", line8e)
	}
	// 32e active+running but stale last-seen -> not alive.
	if !strings.Contains(line32e, "no") {
		t.Errorf("32e should be alive=no (stale): %q", line32e)
	}
}

func TestAliveLabel(t *testing.T) {
	tests := []struct {
		name string
		ws   metrics.WorkerStatus
		want string
	}{
		{"unreachable", metrics.WorkerStatus{Reachable: false}, "unreachable"},
		{"reachable active fresh", metrics.WorkerStatus{Reachable: true, UnitActive: true, LastSeen: 5 * time.Second}, "yes"},
		{"reachable active fresh within window", metrics.WorkerStatus{Reachable: true, UnitActive: true, LastSeen: 90 * time.Second}, "yes"},
		{"reachable active stale", metrics.WorkerStatus{Reachable: true, UnitActive: true, LastSeen: 6 * time.Minute}, "no"},
		{"reachable inactive", metrics.WorkerStatus{Reachable: true, UnitActive: false, LastSeen: 1 * time.Second}, "no"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := aliveLabel(tt.ws); got != tt.want {
				t.Errorf("aliveLabel = %q, want %q", got, tt.want)
			}
		})
	}
}
