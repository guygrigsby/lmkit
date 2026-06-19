// Package unit renders a worker's persistent systemd --user unit file from a
// manifest run. The rendered text is written to
// ~/.config/systemd/user/lmkit-<project>-<run>.service on the box by `lmkit
// run`.
//
// Tilde handling: systemd does NOT expand a leading ~ in unit directives, and
// the CLI cannot know the box's $HOME locally. So a leading "~/" (or a bare
// "~") in workdir or venv is rewritten to systemd's %h specifier, which the
// box's systemd expands to the user's home at load time. Absolute paths and
// ~user forms are emitted verbatim.
package unit

import (
	"fmt"
	"sort"
	"strings"

	"github.com/guygrigsby/lmkit/internal/fleet"
)

// expandTilde rewrites a leading "~/" or a bare "~" to systemd's %h specifier.
// Anything else (absolute, relative, or ~user) is returned unchanged.
func expandTilde(p string) string {
	if p == "~" {
		return "%h"
	}
	if strings.HasPrefix(p, "~/") {
		return "%h" + p[1:]
	}
	return p
}

// Render produces the persistent systemd --user unit text for a run. cmd[0] is
// resolved against {venv}/bin (e.g. "python ..." -> "{venv}/bin/python ...").
// The GPU env line comes from fleet.GPUEnv; any manifest env entries follow in
// sorted key order for deterministic output.
//
// gpuWrap is an optional, generic exec-wrapper template (from the box's
// gpu_wrap). When non-empty it is prepended to ExecStart, with {label} replaced
// by the unit name and {gpu} by the run's normalized GPU id (e.g. "cuda0"). This
// is how a GPU mutex like gputex is hooked in WITHOUT lmkit knowing about it:
// the lock is acquired when systemd starts the unit and released when the
// wrapped process exits. Empty gpuWrap = bare ExecStart (no wrapper, not forced).
func Render(project string, r fleet.Run, gpuWrap string) (string, error) {
	gpuEnv, err := fleet.GPUEnv(r.GPU)
	if err != nil {
		return "", err
	}

	fields := strings.Fields(r.Cmd)
	if len(fields) == 0 {
		return "", fmt.Errorf("run %s/%s: empty cmd", project, r.Name)
	}
	venv := expandTilde(r.Venv)
	execStart := venv + "/bin/" + fields[0]
	if len(fields) > 1 {
		execStart += " " + strings.Join(fields[1:], " ")
	}
	if w := strings.TrimSpace(gpuWrap); w != "" {
		label := "lmkit-" + project + "-" + r.Name
		gpuID := strings.Replace(r.GPU, ":", "", 1) // "cuda:0" -> "cuda0"
		w = strings.NewReplacer("{label}", label, "{gpu}", gpuID).Replace(w)
		w = expandTilde(w) // a leading ~/ (e.g. ~/go/bin/gputex) -> %h; systemd --user PATH won't find a bare binary
		execStart = w + " " + execStart
	}

	var b strings.Builder
	fmt.Fprintf(&b, "[Unit]\n")
	fmt.Fprintf(&b, "Description=lmkit %s/%s\n", project, r.Name)
	fmt.Fprintf(&b, "After=default.target\n")
	fmt.Fprintf(&b, "\n")
	fmt.Fprintf(&b, "[Service]\n")
	fmt.Fprintf(&b, "WorkingDirectory=%s\n", expandTilde(r.Workdir))
	fmt.Fprintf(&b, "Environment=%s\n", gpuEnv)
	for _, k := range sortedKeys(r.Env) {
		fmt.Fprintf(&b, "Environment=%s=%s\n", k, r.Env[k])
	}
	fmt.Fprintf(&b, "ExecStart=%s\n", execStart)
	fmt.Fprintf(&b, "Restart=on-failure\n")
	fmt.Fprintf(&b, "RestartSec=30\n")
	fmt.Fprintf(&b, "TimeoutStopSec=120\n")
	fmt.Fprintf(&b, "\n")
	fmt.Fprintf(&b, "[Install]\n")
	fmt.Fprintf(&b, "WantedBy=default.target\n")
	return b.String(), nil
}

func sortedKeys(m map[string]string) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
