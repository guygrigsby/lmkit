// Package fleet parses the fleet config (~/.config/lmkit/fleet.toml) and the
// per-project manifest (lmkit.toml), and maps a manifest gpu spec to the
// corresponding visibility env-var assignment.
package fleet

import (
	"fmt"

	"github.com/BurntSushi/toml"
)

// Box is one entry under [box.<name>] in fleet.toml: an ssh alias/host.
type Box struct {
	SSH string `toml:"ssh"`
}

// Fleet is the set of boxes declared in fleet.toml, keyed by box name.
type Fleet struct {
	Box map[string]Box `toml:"box"`
}

// LoadFleet reads and validates a fleet.toml at path.
func LoadFleet(path string) (Fleet, error) {
	var f Fleet
	if _, err := toml.DecodeFile(path, &f); err != nil {
		return Fleet{}, fmt.Errorf("load fleet %q: %w", path, err)
	}
	if len(f.Box) == 0 {
		return Fleet{}, fmt.Errorf("load fleet %q: no boxes defined", path)
	}
	for name, b := range f.Box {
		if b.SSH == "" {
			return Fleet{}, fmt.Errorf("load fleet %q: box %q missing ssh", path, name)
		}
	}
	return f, nil
}

// Run is one worker declaration ([[run]]) in a manifest.
type Run struct {
	Name    string            `toml:"name"`
	Box     string            `toml:"box"`
	Venv    string            `toml:"venv"`
	GPU     string            `toml:"gpu"`
	Workdir string            `toml:"workdir"`
	OutDir  string            `toml:"out_dir"`
	Cmd     string            `toml:"cmd"`
	Env     map[string]string `toml:"env"`
}

// Manifest is a per-project lmkit.toml: a project name and its worker runs.
type Manifest struct {
	Project string `toml:"project"`
	Run     []Run  `toml:"run"`
}

// LoadManifest reads and validates a manifest (lmkit.toml) at path. Every run's
// required fields must be present, and its gpu spec must be well-formed.
func LoadManifest(path string) (Manifest, error) {
	var m Manifest
	if _, err := toml.DecodeFile(path, &m); err != nil {
		return Manifest{}, fmt.Errorf("load manifest %q: %w", path, err)
	}
	if m.Project == "" {
		return Manifest{}, fmt.Errorf("load manifest %q: missing project", path)
	}
	if len(m.Run) == 0 {
		return Manifest{}, fmt.Errorf("load manifest %q: no runs defined", path)
	}
	for i, r := range m.Run {
		if err := validateRun(r); err != nil {
			return Manifest{}, fmt.Errorf("load manifest %q: run %d: %w", path, i, err)
		}
	}
	return m, nil
}

func validateRun(r Run) error {
	missing := []struct {
		name, val string
	}{
		{"name", r.Name},
		{"box", r.Box},
		{"venv", r.Venv},
		{"gpu", r.GPU},
		{"workdir", r.Workdir},
		{"out_dir", r.OutDir},
		{"cmd", r.Cmd},
	}
	for _, f := range missing {
		if f.val == "" {
			return fmt.Errorf("missing %s", f.name)
		}
	}
	if _, err := GPUEnv(r.GPU); err != nil {
		return err
	}
	return nil
}
