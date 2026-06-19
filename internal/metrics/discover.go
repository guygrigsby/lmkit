package metrics

import (
	"strings"
	"sync"

	"github.com/guygrigsby/lmkit/internal/remote"
)

// DiscoverBox finds every deployed worker on one box without a manifest: it
// lists metrics.jsonl files under runsRoot over ssh, derives (project, run) from
// each path, and folds each worker's unit state + metrics tail into a
// WorkerStatus. runsRoot is passed literally (e.g. "~/projects/training") so the
// remote login shell expands ~ and the glob; it is never expanded locally.
//
// Reachability is told apart from "no workers" by the ls result: an ssh
// transport failure (ssh's own exit status 255 — connection refused, host
// unreachable) yields a single Reachable=false row; ls exiting non-zero merely
// because the glob matched nothing is reachable-but-empty (an empty slice), as
// is a clean ls with no output.
func DiscoverBox(r remote.Runner, box, sshHost, runsRoot string, now float64) []WorkerStatus {
	glob := runsRoot + "/*/checkpoints*/metrics.jsonl"
	out, err := r.Run(remote.Cmd{
		Host: sshHost,
		Args: []string{"ls", glob},
	})
	if err != nil && isTransportError(err) {
		return []WorkerStatus{{Box: box, Reachable: false}}
	}
	// err != nil without a transport failure (ls found nothing) or a clean exit
	// both fall through: parse whatever paths came back (possibly none).

	var paths []string
	for _, ln := range strings.Split(out, "\n") {
		ln = strings.TrimSpace(ln)
		if ln != "" {
			paths = append(paths, ln)
		}
	}
	if len(paths) == 0 {
		return nil
	}

	statuses := make([]WorkerStatus, len(paths))
	var wg sync.WaitGroup
	for i, path := range paths {
		project, run, ok := runFromMetricsPath(path)
		if !ok {
			continue
		}
		wg.Add(1)
		go func(i int, project, run, path string) {
			defer wg.Done()
			statuses[i] = discoverWorker(r, box, sshHost, project, run, path, now)
		}(i, project, run, path)
	}
	wg.Wait()

	// Drop entries skipped for an unparseable path (zero-value, empty Project).
	res := statuses[:0]
	for _, ws := range statuses {
		if ws.Project != "" {
			res = append(res, ws)
		}
	}
	return res
}

// discoverWorker builds a full WorkerStatus for one discovered worker, reusing
// the same is-active check and metrics-tail fold Assemble uses. The box already
// proved reachable (the ls succeeded), so Reachable is always true here.
func discoverWorker(r remote.Runner, box, sshHost, project, run, path string, now float64) WorkerStatus {
	ws := WorkerStatus{Project: project, Run: run, Box: box, Reachable: true}
	_, active := isActive(r, sshHost, UnitName(project, run))
	ws.UnitActive = active
	foldMetricsInto(&ws, r, sshHost, path, now)
	return ws
}

// runFromMetricsPath derives (project, run) from a metrics.jsonl path of the
// form <runsRoot>/<project>/<ckptdir>/metrics.jsonl. project is the element two
// levels above metrics.jsonl; ckptdir is the element directly above; run is
// ckptdir with a "checkpoints-" prefix stripped, or "default" when ckptdir is
// exactly "checkpoints". It parses the PATH, not the systemd unit name, because
// project and run may contain hyphens (e.g. lm-100m-en). ok is false for any
// path too short to carry those three elements.
func runFromMetricsPath(path string) (project, run string, ok bool) {
	parts := strings.Split(path, "/")
	if len(parts) < 3 {
		return "", "", false
	}
	if parts[len(parts)-1] != "metrics.jsonl" {
		return "", "", false
	}
	ckptdir := parts[len(parts)-2]
	project = parts[len(parts)-3]
	if project == "" || ckptdir == "" {
		return "", "", false
	}
	if ckptdir == "checkpoints" {
		run = "default"
	} else {
		run = strings.TrimPrefix(ckptdir, "checkpoints-")
	}
	return project, run, true
}

// isTransportError reports whether err is an ssh transport failure (ssh's own
// connection failure) rather than a remote command's non-zero exit. ssh reserves
// exit status 255 for its own failures (connection refused, host unreachable,
// auth failure); any other non-zero code is the remote command's. The remote
// package wraps the exit in the error string, so match on "exit status 255".
func isTransportError(err error) bool {
	return err != nil && strings.Contains(err.Error(), "exit status 255")
}
