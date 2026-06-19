package metrics

import (
	"fmt"
	"strings"
	"time"

	"github.com/guygrigsby/lmkit/internal/fleet"
	"github.com/guygrigsby/lmkit/internal/remote"
)

// WorkerStatus is one worker's current state, derived entirely from artifacts
// read over ssh (systemctl is-active + a metrics.jsonl tail). It never reflects
// engine internals.
type WorkerStatus struct {
	Project, Run, Box string
	Step, MaxSteps    int64
	TrainLoss         float64
	TokPerSec         float64
	PeakVRAMGB        float64
	LastSeen          time.Duration // now - last metric ts (box clock)
	UnitActive        bool          // systemctl --user is-active == "active"
	Running           bool          // metrics last-event heuristic
	Reachable         bool          // ssh transport reached the box
}

// systemd is-active stdout values that prove the box answered even though the
// command exited non-zero (the unit is simply not active). Their presence
// means "reachable but not active", NOT "unreachable".
var systemdInactiveStates = map[string]bool{
	"inactive":     true,
	"failed":       true,
	"activating":   true,
	"deactivating": true,
	"reloading":    true,
}

// UnitName is the persistent systemd --user unit backing a worker.
func UnitName(project, run string) string {
	return fmt.Sprintf("lmkit-%s-%s", project, run)
}

// Assemble produces a WorkerStatus for one run by issuing two read-only ssh
// commands through r: `systemctl --user is-active <unit>` and a tail of the
// run's metrics.jsonl. now is the current unix time (seconds) used to compute
// LastSeen against the latest metric ts.
//
// Reachability: ssh transport failure (the box could not be reached at all)
// sets Reachable=false. A box that answers but whose unit is inactive or whose
// metrics file is missing is Reachable=true with UnitActive/Running reflecting
// the real state. The two are told apart by inspecting what the commands
// returned: a non-zero is-active that still printed a known systemd state
// ("inactive"/"failed"/...) means the box answered; an empty stdout with an
// error means the transport itself failed.
func Assemble(project string, r fleet.Run, sshHost string, runner remote.Runner, now float64) WorkerStatus {
	ws := WorkerStatus{
		Project: project,
		Run:     r.Name,
		Box:     r.Box,
	}

	unit := UnitName(project, r.Name)
	reachable, active := isActive(runner, sshHost, unit)
	if !reachable {
		// Empty/unrecognized stdout with an error: the transport failed.
		ws.Reachable = false
		return ws
	}
	ws.Reachable = true
	ws.UnitActive = active

	// Read the metrics tail. A missing file fails non-zero but that is not a
	// transport failure: the box already proved reachable above.
	path := fmt.Sprintf("%s/%s/metrics.jsonl", r.Workdir, r.OutDir)
	foldMetricsInto(&ws, runner, sshHost, path, now)
	return ws
}

// isActive issues `systemctl --user is-active <unit>` over ssh and decides
// reachability + active state from the result. A clean exit ("active", code 0)
// or a non-zero exit whose stdout is a recognized systemd state
// ("inactive"/"failed"/...) both mean the box answered (reachable=true); only an
// empty/unrecognized stdout with an error means the transport itself failed.
func isActive(runner remote.Runner, sshHost, unit string) (reachable, active bool) {
	out, err := runner.Run(remote.Cmd{
		Host: sshHost,
		Args: []string{"systemctl", "--user", "is-active", unit},
	})
	state := strings.TrimSpace(out)
	switch {
	case err == nil:
		return true, state == "active"
	case systemdInactiveStates[state]:
		return true, false
	default:
		return false, false
	}
}

// foldMetricsInto reads the metrics.jsonl tail at path over ssh and folds it
// into ws (Step/MaxSteps/TrainLoss/TokPerSec/PeakVRAMGB/Running/LastSeen). A
// missing or empty file (or unparseable tail) leaves ws unchanged: the caller
// has already set Reachable. ssh joins argv into one string run by the remote
// login shell, which expands a leading ~/ in path — so pass plain args, NOT an
// `sh -c "..."` wrapper (that extra layer mangles the quoting and yields empty
// output).
func foldMetricsInto(ws *WorkerStatus, runner remote.Runner, sshHost, path string, now float64) {
	tail, err := runner.Run(remote.Cmd{
		Host: sshHost,
		Args: []string{"tail", "-c", "65536", path},
	})
	if err != nil || strings.TrimSpace(tail) == "" {
		return
	}
	folded, ok := Fold(tail)
	if !ok {
		return
	}
	ws.Step = folded.Step
	ws.MaxSteps = folded.MaxSteps
	ws.TrainLoss = folded.TrainLoss
	ws.TokPerSec = folded.TokPerSec
	ws.PeakVRAMGB = folded.PeakVRAMGB
	ws.Running = folded.Running
	if folded.TS > 0 {
		ws.LastSeen = time.Duration((now - folded.TS) * float64(time.Second))
	}
}
