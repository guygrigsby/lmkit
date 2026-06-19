package remote

import (
	"bytes"
	"fmt"
	"os/exec"
	"strings"
)

// SSHRunner implements Runner by exec'ing the system ssh binary. bin and argFn
// are fields (not constants) so tests can substitute a stub binary/argv and
// exercise the exec, capture, and error-wrapping paths without a real ssh.
type SSHRunner struct {
	bin   string
	argFn func(Cmd) []string
}

// NewSSHRunner returns an SSHRunner that invokes the system ssh binary.
func NewSSHRunner() *SSHRunner {
	return &SSHRunner{
		bin:   "ssh",
		argFn: sshArgs,
	}
}

// Run executes the Cmd on its host over ssh, returning captured stdout. On a
// non-zero exit it returns an error that includes the captured stderr so the
// remote failure is visible to the caller.
func (r *SSHRunner) Run(c Cmd) (string, error) {
	var stdout, stderr bytes.Buffer
	cmd := exec.Command(r.bin, r.argFn(c)...)
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderr.String())
		if msg == "" {
			return stdout.String(), fmt.Errorf("ssh %s: %w", c.Host, err)
		}
		return stdout.String(), fmt.Errorf("ssh %s: %w: %s", c.Host, err, msg)
	}
	return stdout.String(), nil
}
