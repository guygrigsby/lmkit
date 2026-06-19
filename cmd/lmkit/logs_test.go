package main

import (
	"reflect"
	"testing"
)

func TestParseWorker(t *testing.T) {
	tests := []struct {
		name            string
		arg             string
		wantProj, wantR string
		wantErr         bool
	}{
		{"project and run", "moe/16e", "moe", "16e", false},
		{"run with slash-free name", "anneal/cp2", "anneal", "cp2", false},
		{"missing run", "moe", "", "", true},
		{"empty", "", "", "", true},
		{"trailing slash", "moe/", "", "", true},
		{"leading slash", "/16e", "", "", true},
		{"too many parts", "moe/16e/extra", "", "", true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			proj, run, err := parseWorker(tt.arg)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("parseWorker(%q) = (%q,%q,nil), want error", tt.arg, proj, run)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseWorker(%q) unexpected err: %v", tt.arg, err)
			}
			if proj != tt.wantProj || run != tt.wantR {
				t.Fatalf("parseWorker(%q) = (%q,%q), want (%q,%q)", tt.arg, proj, run, tt.wantProj, tt.wantR)
			}
		})
	}
}

func TestJournalctlArgs(t *testing.T) {
	tests := []struct {
		name   string
		unit   string
		follow bool
		want   []string
	}{
		{
			name:   "non-follow defaults to last 200 lines",
			unit:   "lmkit-moe-16e",
			follow: false,
			want:   []string{"journalctl", "--user", "-u", "lmkit-moe-16e", "-n", "200"},
		},
		{
			name:   "follow streams live",
			unit:   "lmkit-moe-16e",
			follow: true,
			want:   []string{"journalctl", "--user", "-u", "lmkit-moe-16e", "-f"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := journalctlArgs(tt.unit, tt.follow)
			if !reflect.DeepEqual(got, tt.want) {
				t.Fatalf("journalctlArgs(%q, %v) = %v, want %v", tt.unit, tt.follow, got, tt.want)
			}
		})
	}
}
