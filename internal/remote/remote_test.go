package remote

import (
	"reflect"
	"testing"
)

func TestSSHArgs(t *testing.T) {
	tests := []struct {
		name string
		cmd  Cmd
		want []string
	}{
		{
			name: "single arg",
			cmd:  Cmd{Host: "trig", Args: []string{"hostname"}},
			want: []string{"trig", "--", "hostname"},
		},
		{
			name: "multiple args",
			cmd:  Cmd{Host: "bee", Args: []string{"systemctl", "--user", "is-active", "lmkit-moe-16e"}},
			want: []string{"bee", "--", "systemctl", "--user", "is-active", "lmkit-moe-16e"},
		},
		{
			name: "no args",
			cmd:  Cmd{Host: "trig"},
			want: []string{"trig", "--"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := sshArgs(tt.cmd)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("sshArgs(%+v) = %v, want %v", tt.cmd, got, tt.want)
			}
		})
	}
}

// fakeRunner is a test double for Runner, recording the calls it receives and
// returning canned results. It must live only in test code.
type fakeRunner struct {
	calls  []Cmd
	stdout string
	err    error
}

func (f *fakeRunner) Run(c Cmd) (string, error) {
	f.calls = append(f.calls, c)
	return f.stdout, f.err
}

func TestFakeRunnerSatisfiesInterface(t *testing.T) {
	var r Runner = &fakeRunner{stdout: "active\n"}
	out, err := r.Run(Cmd{Host: "trig", Args: []string{"echo", "hi"}})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if out != "active\n" {
		t.Fatalf("got stdout %q, want %q", out, "active\n")
	}
	fr := r.(*fakeRunner)
	if len(fr.calls) != 1 || fr.calls[0].Host != "trig" {
		t.Fatalf("call not recorded: %+v", fr.calls)
	}
}

// TestSSHRunnerErrorWrapsStderr exercises the real SSHRunner against a stubbed
// command path so we verify non-zero exit surfaces an error containing stderr,
// without invoking a real ssh.
func TestSSHRunnerErrorWrapsStderr(t *testing.T) {
	r := NewSSHRunner()
	// Point at a binary that exits non-zero and writes to stderr: `false` has
	// no stderr, so use `sh -c`. We override the binary used by the runner.
	r.bin = "/bin/sh"
	r.argFn = func(c Cmd) []string {
		return []string{"-c", "echo boom 1>&2; exit 3"}
	}
	_, err := r.Run(Cmd{Host: "ignored", Args: []string{"whatever"}})
	if err == nil {
		t.Fatalf("expected error on non-zero exit, got nil")
	}
	if want := "boom"; !contains(err.Error(), want) {
		t.Fatalf("error %q does not contain stderr %q", err.Error(), want)
	}
}

func TestSSHRunnerCapturesStdout(t *testing.T) {
	r := NewSSHRunner()
	r.bin = "/bin/sh"
	r.argFn = func(c Cmd) []string {
		return []string{"-c", "printf 'hello world'"}
	}
	out, err := r.Run(Cmd{Host: "ignored", Args: []string{"whatever"}})
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if out != "hello world" {
		t.Fatalf("got stdout %q, want %q", out, "hello world")
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
