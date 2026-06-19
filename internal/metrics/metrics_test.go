package metrics

import (
	"testing"
)

func TestFold(t *testing.T) {
	tests := []struct {
		name   string
		jsonl  string
		wantOK bool
		check  func(t *testing.T, f Folded)
	}{
		{
			name:   "empty -> not ok",
			jsonl:  "",
			wantOK: false,
		},
		{
			name:   "whitespace only -> not ok",
			jsonl:  "  \n\n   \n",
			wantOK: false,
		},
		{
			name:   "all malformed -> not ok",
			jsonl:  "not json\n{broken\n",
			wantOK: false,
		},
		{
			name: "start+eval+train+train folds to latest values",
			jsonl: `{"event":"start","step":0,"params":124000000,"max_steps":200000,"ts":1000.0}
{"event":"eval","step":1000,"val_loss":2.5,"best_val":2.5,"train_loss":2.6,"lr":0.0006,"improved":true,"ts":1100.0}
{"event":"train","step":1500,"train_loss":2.4,"lr":0.00059,"grad_norm":1.2,"tok_per_sec":42000.0,"step_time_ms":120.0,"tokens_seen":3072000,"tflops":50.0,"peak_vram_gb":18.5,"aux_loss":0.01,"ts":1200.0}
{"event":"train","step":2000,"train_loss":2.3,"lr":0.00058,"tok_per_sec":43000.0,"tokens_seen":4096000,"peak_vram_gb":19.0,"aux_loss":0.02,"ts":1300.0}`,
			wantOK: true,
			check: func(t *testing.T, f Folded) {
				if f.Step != 2000 {
					t.Errorf("Step = %d, want 2000", f.Step)
				}
				if f.MaxSteps != 200000 {
					t.Errorf("MaxSteps = %d, want 200000", f.MaxSteps)
				}
				if f.Params != 124000000 {
					t.Errorf("Params = %d, want 124000000", f.Params)
				}
				if f.TokensSeen != 4096000 {
					t.Errorf("TokensSeen = %d, want 4096000", f.TokensSeen)
				}
				if f.TrainLoss != 2.3 {
					t.Errorf("TrainLoss = %v, want 2.3", f.TrainLoss)
				}
				// val_loss/best_val only ever came from the eval line
				if f.ValLoss != 2.5 {
					t.Errorf("ValLoss = %v, want 2.5", f.ValLoss)
				}
				if f.BestVal != 2.5 {
					t.Errorf("BestVal = %v, want 2.5", f.BestVal)
				}
				if f.LR != 0.00058 {
					t.Errorf("LR = %v, want 0.00058", f.LR)
				}
				if f.TokPerSec != 43000.0 {
					t.Errorf("TokPerSec = %v, want 43000", f.TokPerSec)
				}
				if f.PeakVRAMGB != 19.0 {
					t.Errorf("PeakVRAMGB = %v, want 19.0", f.PeakVRAMGB)
				}
				if f.AuxLoss != 0.02 {
					t.Errorf("AuxLoss = %v, want 0.02", f.AuxLoss)
				}
				if f.TS != 1300.0 {
					t.Errorf("TS = %v, want 1300", f.TS)
				}
				if f.LastEvent != "train" {
					t.Errorf("LastEvent = %q, want train", f.LastEvent)
				}
				if !f.Running {
					t.Errorf("Running = false, want true")
				}
			},
		},
		{
			name: "terminal done sets Running=false",
			jsonl: `{"event":"train","step":2000,"train_loss":2.3,"ts":1300.0}
{"event":"done","step":2000,"best_val":2.1,"ts":1400.0}`,
			wantOK: true,
			check: func(t *testing.T, f Folded) {
				if f.Running {
					t.Errorf("Running = true, want false after done")
				}
				if f.LastEvent != "done" {
					t.Errorf("LastEvent = %q, want done", f.LastEvent)
				}
				if f.BestVal != 2.1 {
					t.Errorf("BestVal = %v, want 2.1", f.BestVal)
				}
			},
		},
		{
			name: "terminal sigterm sets Running=false",
			jsonl: `{"event":"train","step":500,"train_loss":3.0,"ts":900.0}
{"event":"sigterm","step":500,"ts":950.0}`,
			wantOK: true,
			check: func(t *testing.T, f Folded) {
				if f.Running {
					t.Errorf("Running = true, want false after sigterm")
				}
			},
		},
		{
			name: "terminal nan sets Running=false",
			jsonl: `{"event":"train","step":500,"train_loss":3.0,"ts":900.0}
{"event":"nan","step":501,"loss":null,"ts":960.0}`,
			wantOK: true,
			check: func(t *testing.T, f Folded) {
				if f.Running {
					t.Errorf("Running = true, want false after nan")
				}
				if f.LastEvent != "nan" {
					t.Errorf("LastEvent = %q, want nan", f.LastEvent)
				}
			},
		},
		{
			name: "malformed lines skipped, latest valid wins",
			jsonl: `{"event":"start","step":0,"max_steps":100,"ts":1.0}
garbage here
{"event":"train","step":50,"train_loss":1.5,"ts":2.0}

{bad
{"event":"resume","step":60,"ts":3.0}`,
			wantOK: true,
			check: func(t *testing.T, f Folded) {
				if f.Step != 60 {
					t.Errorf("Step = %d, want 60 (resume latest carries step)", f.Step)
				}
				if f.TrainLoss != 1.5 {
					t.Errorf("TrainLoss = %v, want 1.5 (latest carrying line)", f.TrainLoss)
				}
				if f.MaxSteps != 100 {
					t.Errorf("MaxSteps = %d, want 100", f.MaxSteps)
				}
				if f.LastEvent != "resume" {
					t.Errorf("LastEvent = %q, want resume", f.LastEvent)
				}
				if !f.Running {
					t.Errorf("Running = false, want true")
				}
			},
		},
		{
			name: "fields not present on a line do not clobber earlier values",
			jsonl: `{"event":"eval","step":100,"val_loss":2.0,"best_val":2.0,"ts":10.0}
{"event":"train","step":110,"train_loss":1.9,"ts":11.0}`,
			wantOK: true,
			check: func(t *testing.T, f Folded) {
				if f.ValLoss != 2.0 {
					t.Errorf("ValLoss = %v, want 2.0 (kept from eval)", f.ValLoss)
				}
				if f.TrainLoss != 1.9 {
					t.Errorf("TrainLoss = %v, want 1.9", f.TrainLoss)
				}
				if f.Step != 110 {
					t.Errorf("Step = %d, want 110", f.Step)
				}
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, ok := Fold(tt.jsonl)
			if ok != tt.wantOK {
				t.Fatalf("Fold ok = %v, want %v", ok, tt.wantOK)
			}
			if tt.check != nil {
				tt.check(t, f)
			}
		})
	}
}
