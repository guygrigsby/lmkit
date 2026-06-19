package unit

import (
	"strings"
	"testing"

	"github.com/guygrigsby/lmkit/internal/fleet"
)

// The moe 16e run from the design spec, rendered to its exact unit text. A
// leading ~/ in workdir/venv becomes the systemd %h specifier (systemd does
// NOT expand ~). cmd[0] (python) is resolved against {venv}/bin.
func TestRenderMoe16eGolden(t *testing.T) {
	r := fleet.Run{
		Name:    "16e",
		Box:     "trig",
		Venv:    "~/venvs/rocm",
		GPU:     "rocm:0",
		Workdir: "~/projects/training/moe",
		OutDir:  "checkpoints-16e",
		Cmd:     "python train.py --n-experts 16 --out-dir checkpoints-16e --max-steps 200000",
	}
	got, err := Render("moe", r, "")
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	want := `[Unit]
Description=lmkit moe/16e
After=default.target

[Service]
WorkingDirectory=%h/projects/training/moe
Environment=HIP_VISIBLE_DEVICES=0
ExecStart=%h/venvs/rocm/bin/python train.py --n-experts 16 --out-dir checkpoints-16e --max-steps 200000
Restart=on-failure
RestartSec=30
TimeoutStopSec=120

[Install]
WantedBy=default.target
`
	if got != want {
		t.Fatalf("unit mismatch\n--- got ---\n%s\n--- want ---\n%s", got, want)
	}
}

func TestRenderGPUEnvCuda(t *testing.T) {
	r := fleet.Run{
		Name: "8e", Box: "trig", Venv: "/opt/venv", GPU: "cuda:1",
		Workdir: "/srv/moe", OutDir: "checkpoints-8e",
		Cmd: "python train.py",
	}
	got, err := Render("moe", r, "")
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if !strings.Contains(got, "Environment=CUDA_VISIBLE_DEVICES=1") {
		t.Fatalf("missing cuda env line:\n%s", got)
	}
	// Absolute paths must be emitted verbatim (no %h substitution).
	if !strings.Contains(got, "WorkingDirectory=/srv/moe") {
		t.Fatalf("absolute workdir mangled:\n%s", got)
	}
	if !strings.Contains(got, "ExecStart=/opt/venv/bin/python train.py") {
		t.Fatalf("absolute venv ExecStart mangled:\n%s", got)
	}
}

func TestRenderManifestEnvSortedDeterministic(t *testing.T) {
	r := fleet.Run{
		Name: "16e", Box: "trig", Venv: "~/venvs/rocm", GPU: "rocm:0",
		Workdir: "~/m", OutDir: "out", Cmd: "python train.py",
		Env: map[string]string{"ZED": "1", "AIM_REPO": "/aim"},
	}
	got, err := Render("moe", r, "")
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	// GPU env first, then manifest env in sorted key order, deterministically.
	gpuIdx := strings.Index(got, "Environment=HIP_VISIBLE_DEVICES=0")
	aimIdx := strings.Index(got, "Environment=AIM_REPO=/aim")
	zedIdx := strings.Index(got, "Environment=ZED=1")
	if gpuIdx < 0 || aimIdx < 0 || zedIdx < 0 {
		t.Fatalf("missing an env line:\n%s", got)
	}
	if !(gpuIdx < aimIdx && aimIdx < zedIdx) {
		t.Fatalf("env lines not in gpu,sorted order:\n%s", got)
	}
}

func TestRenderRejectsBadGPU(t *testing.T) {
	r := fleet.Run{
		Name: "x", Box: "trig", Venv: "~/v", GPU: "tpu:0",
		Workdir: "~/m", OutDir: "out", Cmd: "python train.py",
	}
	if _, err := Render("moe", r, ""); err == nil {
		t.Fatalf("expected error for bad gpu spec")
	}
}

// A non-empty gpu_wrap is prepended to ExecStart with {gpu}/{label} substituted;
// an empty wrap leaves ExecStart bare (the wrapper is opt-in, never forced).
func TestRenderGpuWrap(t *testing.T) {
	r := fleet.Run{
		Name: "16e", Box: "trig", Venv: "~/venvs/rocm", GPU: "rocm:0",
		Workdir: "~/projects/training/moe", OutDir: "checkpoints-16e",
		Cmd: "python train.py --n-experts 16",
	}
	got, err := Render("moe", r, `gputex run --gpu {gpu} "{label}" --`)
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	want := `ExecStart=gputex run --gpu rocm0 "lmkit-moe-16e" -- %h/venvs/rocm/bin/python train.py --n-experts 16`
	if !strings.Contains(got, want) {
		t.Errorf("ExecStart not wrapped.\ngot:\n%s\nwant line:\n%s", got, want)
	}
	bare, err := Render("moe", r, "")
	if err != nil {
		t.Fatalf("Render bare: %v", err)
	}
	if strings.Contains(bare, "gputex") {
		t.Errorf("empty wrap must not add a prefix:\n%s", bare)
	}
}

func TestRenderRejectsEmptyCmd(t *testing.T) {
	r := fleet.Run{
		Name: "x", Box: "trig", Venv: "~/v", GPU: "rocm:0",
		Workdir: "~/m", OutDir: "out", Cmd: "   ",
	}
	if _, err := Render("moe", r, ""); err == nil {
		t.Fatalf("expected error for empty cmd")
	}
}

func TestExpandTilde(t *testing.T) {
	tests := []struct{ in, want string }{
		{"~/venvs/rocm", "%h/venvs/rocm"},
		{"~", "%h"},
		{"/opt/venv", "/opt/venv"},
		{"relative/path", "relative/path"},
		{"~foo/bar", "~foo/bar"}, // ~user is not ours to expand
	}
	for _, tt := range tests {
		if got := expandTilde(tt.in); got != tt.want {
			t.Errorf("expandTilde(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}
