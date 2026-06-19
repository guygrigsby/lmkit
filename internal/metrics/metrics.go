// Package metrics turns a worker's metrics.jsonl tail (fetched over ssh) into a
// folded status. The JSONL event schema lives only here: the CLI never imports
// or executes the training engine to learn status, it reads the artifacts the
// engine writes. This mirrors the Python reference fold (latest-value-wins per
// field) rather than importing it.
package metrics

import (
	"encoding/json"
	"strings"
)

// Folded is the result of folding a metrics.jsonl tail: for each field, the
// value from the LATEST line that carried it. Zero values are fine for fields
// no line ever set.
type Folded struct {
	Step       int64
	MaxSteps   int64
	Params     int64
	TokensSeen int64
	TrainLoss  float64
	ValLoss    float64
	BestVal    float64
	LR         float64
	TokPerSec  float64
	PeakVRAMGB float64
	AuxLoss    float64
	TS         float64
	LastEvent  string
	Running    bool
}

// terminalEvents are the last-event values that mean the worker is no longer
// running (it finished, was signalled to stop, or diverged).
var terminalEvents = map[string]bool{
	"done":    true,
	"sigterm": true,
	"nan":     true,
}

// line is the decode target for one metrics.jsonl record. Numeric fields are
// *pointers so an absent field (nil) is distinguishable from a present zero,
// which is what lets a later line that omits a field leave the earlier value
// untouched (latest-value-wins, not latest-line-wins).
type line struct {
	Event      *string  `json:"event"`
	TS         *float64 `json:"ts"`
	Step       *int64   `json:"step"`
	MaxSteps   *int64   `json:"max_steps"`
	Params     *int64   `json:"params"`
	TokensSeen *int64   `json:"tokens_seen"`
	TrainLoss  *float64 `json:"train_loss"`
	ValLoss    *float64 `json:"val_loss"`
	BestVal    *float64 `json:"best_val"`
	LR         *float64 `json:"lr"`
	TokPerSec  *float64 `json:"tok_per_sec"`
	PeakVRAMGB *float64 `json:"peak_vram_gb"`
	AuxLoss    *float64 `json:"aux_loss"`
}

// Fold parses a metrics.jsonl tail and folds it: each field takes the value
// from the latest line that carried it. Malformed and empty lines are skipped.
// Running is false iff the last (non-skipped) event is terminal. The bool is
// false when no line parsed at all.
func Fold(jsonl string) (Folded, bool) {
	var f Folded
	any := false
	for _, raw := range strings.Split(jsonl, "\n") {
		raw = strings.TrimSpace(raw)
		if raw == "" {
			continue
		}
		var l line
		if err := json.Unmarshal([]byte(raw), &l); err != nil {
			continue
		}
		any = true
		if l.Event != nil {
			f.LastEvent = *l.Event
		}
		if l.TS != nil {
			f.TS = *l.TS
		}
		if l.Step != nil {
			f.Step = *l.Step
		}
		if l.MaxSteps != nil {
			f.MaxSteps = *l.MaxSteps
		}
		if l.Params != nil {
			f.Params = *l.Params
		}
		if l.TokensSeen != nil {
			f.TokensSeen = *l.TokensSeen
		}
		if l.TrainLoss != nil {
			f.TrainLoss = *l.TrainLoss
		}
		if l.ValLoss != nil {
			f.ValLoss = *l.ValLoss
		}
		if l.BestVal != nil {
			f.BestVal = *l.BestVal
		}
		if l.LR != nil {
			f.LR = *l.LR
		}
		if l.TokPerSec != nil {
			f.TokPerSec = *l.TokPerSec
		}
		if l.PeakVRAMGB != nil {
			f.PeakVRAMGB = *l.PeakVRAMGB
		}
		if l.AuxLoss != nil {
			f.AuxLoss = *l.AuxLoss
		}
	}
	if !any {
		return Folded{}, false
	}
	f.Running = !terminalEvents[f.LastEvent]
	return f, true
}
