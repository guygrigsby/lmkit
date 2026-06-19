package main

import (
	"fmt"
	"strings"

	"github.com/guygrigsby/lmkit/internal/fleet"
	"github.com/guygrigsby/lmkit/internal/metrics"
	"github.com/guygrigsby/lmkit/internal/remote"
)

// parseWorker splits a "project/run" argument into its two parts. Both parts
// must be non-empty and there must be exactly one slash.
func parseWorker(arg string) (project, run string, err error) {
	project, run, ok := strings.Cut(arg, "/")
	if !ok || project == "" || run == "" {
		return "", "", fmt.Errorf("worker %q: want project/run", arg)
	}
	if strings.Contains(run, "/") {
		return "", "", fmt.Errorf("worker %q: want project/run", arg)
	}
	return project, run, nil
}

// resolved is a worker located in a manifest and mapped to its box's ssh host.
type resolved struct {
	project string
	run     fleet.Run
	sshHost string
	gpuWrap string
}

// resolveWorker loads the manifest and fleet, finds the named run in the
// manifest, and resolves its box to an ssh host. Shared by every command that
// addresses a single worker (logs/run/start/stop).
func resolveWorker(manifestPath, project, run string) (resolved, error) {
	man, err := fleet.LoadManifest(manifestPath)
	if err != nil {
		return resolved{}, err
	}
	if man.Project != project {
		return resolved{}, fmt.Errorf("manifest project is %q, not %q", man.Project, project)
	}
	flt, err := fleet.LoadFleet(fleetConfigPath())
	if err != nil {
		return resolved{}, err
	}
	for _, r := range man.Run {
		if r.Name != run {
			continue
		}
		box, ok := flt.Box[r.Box]
		if !ok {
			return resolved{}, fmt.Errorf("box %q (run %s/%s) not in fleet config", r.Box, project, run)
		}
		return resolved{project: project, run: r, sshHost: box.SSH, gpuWrap: box.GpuWrap}, nil
	}
	return resolved{}, fmt.Errorf("run %q not found in manifest for project %q", run, project)
}

// journalctlArgs builds the remote argv for viewing a unit's journal. Without
// -f it shows the last 200 lines and exits; with -f it follows live. Pure
// function so the argv is unit-tested without invoking ssh.
func journalctlArgs(unit string, follow bool) []string {
	args := []string{"journalctl", "--user", "-u", unit}
	if follow {
		return append(args, "-f")
	}
	return append(args, "-n", "200")
}

func runLogs(args []string) error {
	fs := newFlagSet("logs")
	manifestPath := fs.String("manifest", "lmkit.toml", "path to the project manifest")
	follow := fs.Bool("f", false, "stream the journal live (journalctl -f)")
	pos, err := parseArgs(fs, args)
	if err != nil {
		return err
	}
	if len(pos) != 1 {
		return fmt.Errorf("usage: lmkit logs <project/run> [-f]")
	}
	project, run, err := parseWorker(pos[0])
	if err != nil {
		return err
	}
	w, err := resolveWorker(*manifestPath, project, run)
	if err != nil {
		return err
	}

	unit := metrics.UnitName(w.project, w.run.Name)
	cmd := remote.Cmd{Host: w.sshHost, Args: journalctlArgs(unit, *follow)}

	// -f streams live to the terminal (so the follow output appears in real
	// time and Ctrl-C reaches the remote follower); without -f the bounded
	// output is fine to stream too, which keeps line ordering identical.
	return remote.NewSSHRunner().Stream(cmd)
}
