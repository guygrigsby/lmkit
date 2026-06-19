// Command lmkit is the local-only fleet ops CLI for lmkit training workers.
// ssh is its sole transport; it never opens HTTP and never imports the training
// engine. This phase implements the read-only `status` command.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	"github.com/guygrigsby/lmkit/internal/fleet"
	"github.com/guygrigsby/lmkit/internal/metrics"
	"github.com/guygrigsby/lmkit/internal/remote"
)

const usage = `lmkit - local fleet ops for lmkit training workers

usage:
  lmkit status [--manifest PATH] [--box NAME] [--json]

status reads each worker's systemd unit state and metrics.jsonl tail over ssh
and prints a table (or JSON with --json).
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	switch os.Args[1] {
	case "status":
		if err := runStatus(os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, "lmkit status:", err)
			os.Exit(1)
		}
	default:
		fmt.Fprintf(os.Stderr, "lmkit: unknown command %q\n\n%s", os.Args[1], usage)
		os.Exit(2)
	}
}

// fleetConfigPath returns the path to fleet.toml under the user's config dir,
// falling back to $HOME/.config when os.UserConfigDir is unavailable.
func fleetConfigPath() string {
	if dir, err := os.UserConfigDir(); err == nil && dir != "" {
		return filepath.Join(dir, "lmkit", "fleet.toml")
	}
	return filepath.Join(os.Getenv("HOME"), ".config", "lmkit", "fleet.toml")
}

// newFlagSet returns a flag set for a subcommand whose Usage prints the
// top-level usage string, and which returns errors (rather than exiting) on
// parse failure so the caller controls the exit.
func newFlagSet(name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprint(os.Stderr, usage) }
	return fs
}

func runStatus(args []string) error {
	fs := newFlagSet("status")
	manifestPath := fs.String("manifest", "lmkit.toml", "path to the project manifest")
	boxFilter := fs.String("box", "", "only show workers on this box")
	asJSON := fs.Bool("json", false, "emit JSON instead of a table")
	if err := fs.Parse(args); err != nil {
		return err
	}

	flt, err := fleet.LoadFleet(fleetConfigPath())
	if err != nil {
		return err
	}
	man, err := fleet.LoadManifest(*manifestPath)
	if err != nil {
		return err
	}

	runner := remote.NewSSHRunner()
	now := float64(time.Now().Unix())

	statuses := gatherStatuses(man, flt, runner, now, *boxFilter)

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(statuses)
	}
	fmt.Print(renderTable(statuses))
	return nil
}

// gatherStatuses assembles a WorkerStatus for every run (optionally filtered to
// one box) concurrently: one goroutine per worker, each doing its own ssh. An
// unreachable box yields Reachable=false and never aborts the others. Results
// come back in a stable (manifest) order.
func gatherStatuses(man fleet.Manifest, flt fleet.Fleet, runner remote.Runner, now float64, boxFilter string) []metrics.WorkerStatus {
	type indexed struct {
		i  int
		ws metrics.WorkerStatus
	}
	out := make(chan indexed)
	var wg sync.WaitGroup

	n := 0
	for i, r := range man.Run {
		if boxFilter != "" && r.Box != boxFilter {
			continue
		}
		n++
		wg.Add(1)
		go func(i int, r fleet.Run) {
			defer wg.Done()
			ws := metrics.WorkerStatus{Project: man.Project, Run: r.Name, Box: r.Box}
			box, ok := flt.Box[r.Box]
			if !ok {
				// Box not in the fleet config: treat as unreachable (no ssh host).
				out <- indexed{i, ws}
				return
			}
			out <- indexed{i, metrics.Assemble(man.Project, r, box.SSH, runner, now)}
		}(i, r)
	}

	go func() {
		wg.Wait()
		close(out)
	}()

	res := make([]indexed, 0, n)
	for r := range out {
		res = append(res, r)
	}
	sort.Slice(res, func(a, b int) bool { return res[a].i < res[b].i })
	statuses := make([]metrics.WorkerStatus, len(res))
	for i, r := range res {
		statuses[i] = r.ws
	}
	return statuses
}

// aliveLabel renders the ALIVE column: a worker is alive when its box is
// reachable, its unit is active, and its last metric is recent (< 60s). An
// unreachable box is its own label so it stands apart from a stopped worker.
func aliveLabel(ws metrics.WorkerStatus) string {
	if !ws.Reachable {
		return "unreachable"
	}
	if ws.UnitActive && ws.LastSeen > 0 && ws.LastSeen < 60*time.Second {
		return "yes"
	}
	if ws.UnitActive && ws.LastSeen == 0 {
		// Active but no metric ts yet (just started): treat as alive.
		return "yes"
	}
	return "no"
}

// renderTable formats worker statuses as an aligned text table. Pure function:
// no I/O, so it is unit-tested directly.
func renderTable(rows []metrics.WorkerStatus) string {
	var b strings.Builder
	w := tabwriter.NewWriter(&b, 0, 2, 2, ' ', 0)
	fmt.Fprintln(w, "PROJECT/RUN\tBOX\tSTEP\tPROGRESS%\tLOSS\tTOK/S\tVRAM\tLAST-SEEN\tALIVE")
	for _, ws := range rows {
		worker := fmt.Sprintf("%s/%s", ws.Project, ws.Run)
		if !ws.Reachable {
			fmt.Fprintf(w, "%s\t%s\t-\t-\t-\t-\t-\t-\t%s\n", worker, ws.Box, aliveLabel(ws))
			continue
		}
		progress := "-"
		if ws.MaxSteps > 0 {
			progress = fmt.Sprintf("%.1f%%", float64(ws.Step)/float64(ws.MaxSteps)*100)
		}
		fmt.Fprintf(w, "%s\t%s\t%d\t%s\t%.4g\t%.0f\t%.1f\t%s\t%s\n",
			worker, ws.Box, ws.Step, progress, ws.TrainLoss, ws.TokPerSec, ws.PeakVRAMGB,
			lastSeenLabel(ws.LastSeen), aliveLabel(ws))
	}
	w.Flush()
	return b.String()
}

// lastSeenLabel renders a duration compactly (e.g. "10s", "5m0s"); a zero
// duration (no metric ts read) shows "-".
func lastSeenLabel(d time.Duration) string {
	if d <= 0 {
		return "-"
	}
	return d.Round(time.Second).String()
}
