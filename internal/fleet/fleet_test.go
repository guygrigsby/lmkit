package fleet

import (
	"os"
	"path/filepath"
	"testing"
)

func writeTemp(t *testing.T, name, content string) string {
	t.Helper()
	dir := t.TempDir()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
		t.Fatalf("write temp %s: %v", name, err)
	}
	return p
}

const validFleet = `
[box.trig]
ssh = "trig"
[box.bee]
ssh = "bee"
`

func TestLoadFleet(t *testing.T) {
	p := writeTemp(t, "fleet.toml", validFleet)
	f, err := LoadFleet(p)
	if err != nil {
		t.Fatalf("LoadFleet: %v", err)
	}
	if len(f.Box) != 2 {
		t.Fatalf("got %d boxes, want 2", len(f.Box))
	}
	if f.Box["trig"].SSH != "trig" {
		t.Errorf("trig ssh = %q, want trig", f.Box["trig"].SSH)
	}
	if f.Box["bee"].SSH != "bee" {
		t.Errorf("bee ssh = %q, want bee", f.Box["bee"].SSH)
	}
}

func TestLoadFleetMissingSSH(t *testing.T) {
	p := writeTemp(t, "fleet.toml", "[box.trig]\n")
	if _, err := LoadFleet(p); err == nil {
		t.Fatal("expected error for box missing ssh, got nil")
	}
}

func TestLoadFleetMissingFile(t *testing.T) {
	if _, err := LoadFleet("/no/such/fleet.toml"); err == nil {
		t.Fatal("expected error for missing file, got nil")
	}
}

func TestLoadFleetBadTOML(t *testing.T) {
	p := writeTemp(t, "fleet.toml", "[box.trig\nssh = ")
	if _, err := LoadFleet(p); err == nil {
		t.Fatal("expected error for malformed toml, got nil")
	}
}

// The two-run moe example from the spec.
const validManifest = `
project = "moe"

[[run]]
name    = "16e"
box     = "trig"
venv    = "~/venvs/rocm"
gpu     = "rocm:0"
workdir = "~/projects/training/moe"
out_dir = "checkpoints-16e"
cmd     = "python train.py --n-experts 16 --out-dir checkpoints-16e --max-steps 200000"

[[run]]
name    = "8e"
box     = "trig"
venv    = "~/venvs/cuda"
gpu     = "cuda:0"
workdir = "~/projects/training/moe"
out_dir = "checkpoints-8e"
cmd     = "python train.py --n-experts 8 --out-dir checkpoints-8e --batch-size 2 --grad-accum 32 --max-steps 200000"
`

func TestLoadManifest(t *testing.T) {
	p := writeTemp(t, "lmkit.toml", validManifest)
	m, err := LoadManifest(p)
	if err != nil {
		t.Fatalf("LoadManifest: %v", err)
	}
	if m.Project != "moe" {
		t.Errorf("project = %q, want moe", m.Project)
	}
	if len(m.Run) != 2 {
		t.Fatalf("got %d runs, want 2", len(m.Run))
	}
	r0 := m.Run[0]
	if r0.Name != "16e" || r0.Box != "trig" || r0.Venv != "~/venvs/rocm" ||
		r0.GPU != "rocm:0" || r0.Workdir != "~/projects/training/moe" ||
		r0.OutDir != "checkpoints-16e" {
		t.Errorf("run[0] fields wrong: %+v", r0)
	}
	if r0.Cmd != "python train.py --n-experts 16 --out-dir checkpoints-16e --max-steps 200000" {
		t.Errorf("run[0] cmd wrong: %q", r0.Cmd)
	}
	if m.Run[1].Name != "8e" || m.Run[1].GPU != "cuda:0" {
		t.Errorf("run[1] fields wrong: %+v", m.Run[1])
	}
}

func TestLoadManifestWithEnv(t *testing.T) {
	const manifest = `
project = "moe"
[[run]]
name    = "16e"
box     = "trig"
venv    = "~/venvs/rocm"
gpu     = "rocm:0"
workdir = "~/w"
out_dir = "out"
cmd     = "python train.py"
[run.env]
AIM_REPO = "/mnt/aim"
LMKIT_PROM_TEXTFILE_DIR = "/var/lib/node_exporter"
`
	p := writeTemp(t, "lmkit.toml", manifest)
	m, err := LoadManifest(p)
	if err != nil {
		t.Fatalf("LoadManifest: %v", err)
	}
	env := m.Run[0].Env
	if env["AIM_REPO"] != "/mnt/aim" {
		t.Errorf("AIM_REPO = %q", env["AIM_REPO"])
	}
	if env["LMKIT_PROM_TEXTFILE_DIR"] != "/var/lib/node_exporter" {
		t.Errorf("textfile dir = %q", env["LMKIT_PROM_TEXTFILE_DIR"])
	}
}

func TestLoadManifestMissingProject(t *testing.T) {
	const manifest = `
[[run]]
name = "16e"
box = "trig"
venv = "~/v"
gpu = "rocm:0"
workdir = "~/w"
out_dir = "out"
cmd = "python train.py"
`
	p := writeTemp(t, "lmkit.toml", manifest)
	if _, err := LoadManifest(p); err == nil {
		t.Fatal("expected error for missing project, got nil")
	}
}

func TestLoadManifestMissingRequiredRunField(t *testing.T) {
	// missing cmd
	const manifest = `
project = "moe"
[[run]]
name = "16e"
box = "trig"
venv = "~/v"
gpu = "rocm:0"
workdir = "~/w"
out_dir = "out"
`
	p := writeTemp(t, "lmkit.toml", manifest)
	if _, err := LoadManifest(p); err == nil {
		t.Fatal("expected error for run missing cmd, got nil")
	}
}

func TestLoadManifestNoRuns(t *testing.T) {
	p := writeTemp(t, "lmkit.toml", `project = "moe"`+"\n")
	if _, err := LoadManifest(p); err == nil {
		t.Fatal("expected error for manifest with no runs, got nil")
	}
}

func TestLoadManifestBadGPU(t *testing.T) {
	const manifest = `
project = "moe"
[[run]]
name = "16e"
box = "trig"
venv = "~/v"
gpu = "tpu:0"
workdir = "~/w"
out_dir = "out"
cmd = "python train.py"
`
	p := writeTemp(t, "lmkit.toml", manifest)
	if _, err := LoadManifest(p); err == nil {
		t.Fatal("expected error for unknown gpu vendor, got nil")
	}
}

func TestGPUEnv(t *testing.T) {
	tests := []struct {
		gpu     string
		want    string
		wantErr bool
	}{
		{"rocm:0", "HIP_VISIBLE_DEVICES=0", false},
		{"cuda:1", "CUDA_VISIBLE_DEVICES=1", false},
		{"cuda:0", "CUDA_VISIBLE_DEVICES=0", false},
		{"rocm:3", "HIP_VISIBLE_DEVICES=3", false},
		{"", "", true},
		{"rocm", "", true},     // no colon
		{"tpu:0", "", true},    // unknown vendor
		{"rocm:x", "", true},   // non-numeric index
		{"rocm:", "", true},    // empty index
		{":0", "", true},       // empty vendor
		{"cuda:0:1", "", true}, // too many parts
		{"cuda:-1", "", true},  // negative index
	}
	for _, tt := range tests {
		t.Run(tt.gpu, func(t *testing.T) {
			got, err := GPUEnv(tt.gpu)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("GPUEnv(%q) = %q, want error", tt.gpu, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("GPUEnv(%q): %v", tt.gpu, err)
			}
			if got != tt.want {
				t.Errorf("GPUEnv(%q) = %q, want %q", tt.gpu, got, tt.want)
			}
		})
	}
}
