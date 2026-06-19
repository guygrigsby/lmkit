package main

import "testing"

func TestParseArgs(t *testing.T) {
	cases := []struct {
		name     string
		args     []string
		wantPos  []string
		wantMan  string
		wantFlag bool
	}{
		{"positional only", []string{"foo/bar"}, []string{"foo/bar"}, "lmkit.toml", false},
		{"flag before positional", []string{"--manifest", "x.toml", "foo/bar"}, []string{"foo/bar"}, "x.toml", false},
		{"flag after positional", []string{"foo/bar", "--manifest", "x.toml"}, []string{"foo/bar"}, "x.toml", false},
		{"bool flag after positional", []string{"foo/bar", "-f"}, []string{"foo/bar"}, "lmkit.toml", true},
		{"bool flag before positional", []string{"-f", "foo/bar"}, []string{"foo/bar"}, "lmkit.toml", true},
		{"no positional", []string{"--manifest", "x.toml"}, nil, "x.toml", false},
		{"flags both sides", []string{"-f", "foo/bar", "--manifest", "x.toml"}, []string{"foo/bar"}, "x.toml", true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			fs := newFlagSet("test")
			man := fs.String("manifest", "lmkit.toml", "")
			f := fs.Bool("f", false, "")
			pos, err := parseArgs(fs, c.args)
			if err != nil {
				t.Fatalf("parseArgs: %v", err)
			}
			if len(pos) != len(c.wantPos) || (len(pos) > 0 && pos[0] != c.wantPos[0]) {
				t.Errorf("positionals = %v, want %v", pos, c.wantPos)
			}
			if *man != c.wantMan {
				t.Errorf("manifest = %q, want %q", *man, c.wantMan)
			}
			if *f != c.wantFlag {
				t.Errorf("-f = %v, want %v", *f, c.wantFlag)
			}
		})
	}
}
