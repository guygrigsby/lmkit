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
	out, err := runner.Run(remote.Cmd{
		Host: sshHost,
		Args: []string{"systemctl", "--user", "is-active", unit},
	})
	state := strings.TrimSpace(out)

	switch {
	case err == nil:
		// Clean exit: is-active prints "active" and returns 0.
		ws.Reachable = true
		ws.UnitActive = state == "active"
	case systemdInactiveStates[state]:
		// Non-zero exit but the box clearly answered with a systemd state.
		ws.Reachable = true
		ws.UnitActive = false
	default:
		// Empty/unrecognized stdout with an error: the transport failed.
		ws.Reachable = false
		return ws
	}

	// Read the metrics tail. Run it through a shell so a leading ~/ in workdir
	// expands on the box. A missing file fails non-zero but that is not a
	// transport failure: the box already proved reachable above.
	path := fmt.Sprintf("%s/%s/metrics.jsonl", r.Workdir, r.OutDir)
	tailCmd := fmt.Sprintf("tail -c 65536 %s", path)
	tail, terr := runner.Run(remote.Cmd{
		Host: sshHost,
		Args: []string{"sh", "-c", tailCmd},
	})
	if terr != nil || strings.TrimSpace(tail) == "" {
		return ws
	}

	folded, ok := Fold(tail)
	if !ok {
		return ws
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
	return ws
}
