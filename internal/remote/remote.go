// Package remote owns the sole transport to a box: the system ssh binary,
// invoked via os/exec. It deliberately shells out to ssh (rather than using an
// in-process ssh library) so it inherits the user's ~/.ssh/config host aliases,
// keys, and agent.
package remote

// Cmd is a command to run on a remote box. Args is the argv to execute there.
// Stdin, if non-empty, is fed to the remote process's standard input (used to
// deliver a unit file's contents to a `cat >` on the box).
type Cmd struct {
	Host  string
	Args  []string
	Stdin string
}

// Runner runs a Cmd on its Host and returns captured stdout. Everything
// downstream depends on this interface so it is unit-testable with a fake.
type Runner interface {
	Run(Cmd) (stdout string, err error)
}

// Streamer runs a Cmd with the child's stdio wired straight to the local
// terminal (no buffering) so output appears live and Ctrl-C / `journalctl -f`
// work. Used for `logs -f`; the buffered Run is for everything that needs the
// output as a value.
type Streamer interface {
	Stream(Cmd) error
}

// sshArgs builds the argv passed to the ssh binary for a Cmd:
//
//	ssh <host> -- <args...>
//
// The `--` ends ssh option parsing so the remote argv can never be mistaken for
// local ssh flags. It is a pure function so it can be tested without invoking ssh.
func sshArgs(c Cmd) []string {
	args := make([]string, 0, len(c.Args)+2)
	args = append(args, c.Host, "--")
	args = append(args, c.Args...)
	return args
}
