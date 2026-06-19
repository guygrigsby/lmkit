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
  lmkit status [--box NAME] [--json]
  lmkit logs   <project/run> [--manifest PATH] [-f]
  lmkit run    <project/run | project> [--manifest PATH]
  lmkit start  <project/run> [--manifest PATH] | --all [--box NAME]
  lmkit stop   <project/run> [--manifest PATH] | --all [--box NAME]

status reads each worker's systemd unit state and metrics.jsonl tail over ssh
and prints a table (or JSON with --json).

logs streams a worker's systemd journal over ssh (journalctl --user); -f follows.

run/start/stop are ACID: each takes a per-worker lock, verifies the end state
over ssh, and rolls back on any failure. run deploys (or updates) the
persistent unit and starts it; a bare project runs every worker in the manifest.
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
	case "logs":
		if err := runLogs(os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, "lmkit logs:", err)
			os.Exit(1)
		}
	case "run":
		if err := runRun(os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, "lmkit run:", err)
			os.Exit(1)
		}
	case "start", "stop":
		if err := runStartStop(os.Args[1], os.Args[2:]); err != nil {
			fmt.Fprintln(os.Stderr, "lmkit "+os.Args[1]+":", err)
			os.Exit(1)
		}
	default:
		fmt.Fprintf(os.Stderr, "lmkit: unknown command %q\n\n%s", os.Args[1], usage)
		os.Exit(2)
	}
}

// fleetConfigPath returns the path to fleet.toml under the XDG config dir
// ($XDG_CONFIG_HOME, else ~/.config). Not os.UserConfigDir: on macOS that
// resolves to ~/Library/Application Support, but lmkit config lives in ~/.config.
func fleetConfigPath() string {
	dir := os.Getenv("XDG_CONFIG_HOME")
	if dir == "" {
		dir = filepath.Join(os.Getenv("HOME"), ".config")
	}
	return filepath.Join(dir, "lmkit", "fleet.toml")
}

// newFlagSet returns a flag set for a subcommand whose Usage prints the
// top-level usage string, and which returns errors (rather than exiting) on
// parse failure so the caller controls the exit.
func newFlagSet(name string) *flag.FlagSet {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprint(os.Stderr, usage) }
	return fs
}

// parseArgs parses flags that may appear before OR after positionals. The stdlib
// flag package stops at the first positional, so a natural `lmkit run foo/bar
// --manifest x` would silently drop the flag. This resumes parsing after each
// positional and returns the collected positionals.
func parseArgs(fs *flag.FlagSet, args []string) ([]string, error) {
	var positionals []string
	for {
		if err := fs.Parse(args); err != nil {
			return nil, err
		}
		rest := fs.Args()
		if len(rest) == 0 {
			return positionals, nil
		}
		positionals = append(positionals, rest[0])
		args = rest[1:]
	}
}

func runStatus(args []string) error {
	fs := newFlagSet("status")
	boxFilter := fs.String("box", "", "only show workers on this box")
	asJSON := fs.Bool("json", false, "emit JSON instead of a table")
	if err := fs.Parse(args); err != nil {
		return err
	}

	flt, err := fleet.LoadFleet(fleetConfigPath())
	if err != nil {
		return err
	}

	runner := remote.NewSSHRunner()
	now := float64(time.Now().Unix())

	statuses := gatherStatuses(flt, runner, now, *boxFilter)

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(statuses)
	}
	fmt.Print(renderTable(statuses))
	return nil
}

// gatherStatuses discovers every deployed worker across the fleet (optionally
// filtered to one box) — no manifest. One goroutine per box globs its runs_root
// for metrics.jsonl and resolves each worker over ssh; an unreachable box yields
// a single Reachable=false row and never aborts the others. Rows are sorted by
// box then project/run for stable output.
func gatherStatuses(flt fleet.Fleet, runner remote.Runner, now float64, boxFilter string) []metrics.WorkerStatus {
	out := make(chan []metrics.WorkerStatus)
	var wg sync.WaitGroup
	for name, box := range flt.Box {
		if boxFilter != "" && name != boxFilter {
			continue
		}
		wg.Add(1)
		go func(name string, box fleet.Box) {
			defer wg.Done()
			out <- metrics.DiscoverBox(runner, name, box.SSH, box.RunsRootOr(), now)
		}(name, box)
	}
	go func() {
		wg.Wait()
		close(out)
	}()

	var statuses []metrics.WorkerStatus
	for ws := range out {
		statuses = append(statuses, ws...)
	}
	sort.Slice(statuses, func(a, b int) bool {
		if statuses[a].Box != statuses[b].Box {
			return statuses[a].Box < statuses[b].Box
		}
		if statuses[a].Project != statuses[b].Project {
			return statuses[a].Project < statuses[b].Project
		}
		return statuses[a].Run < statuses[b].Run
	})
	return statuses
}

// freshWindow is how recent the last metric must be for an active unit to read
// "yes". It must exceed the worker's log cadence: metrics log every log_interval
// steps (default 20), which at ~20k tok/s is ~2min between lines, plus eval
// gaps — so 60s would flap a healthy worker to "no". 5min covers the cadence
// with margin; a longer silence on an active unit means it's wedged, not alive.
const freshWindow = 5 * time.Minute

// aliveLabel renders the ALIVE column: a worker is alive when its box is
// reachable, its unit is active, and its last metric is within freshWindow. An
// unreachable box is its own label so it stands apart from a stopped worker.
func aliveLabel(ws metrics.WorkerStatus) string {
	if !ws.Reachable {
		return "unreachable"
	}
	if ws.UnitActive && ws.LastSeen > 0 && ws.LastSeen < freshWindow {
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
