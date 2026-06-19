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
func Render(project string, r fleet.Run) (string, error) {
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
