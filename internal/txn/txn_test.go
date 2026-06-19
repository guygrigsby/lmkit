package txn

import (
	"errors"
	"reflect"
	"strings"
	"testing"
)

// recordStep builds a Step that appends "do:<name>"/"undo:<name>" to log and
// optionally fails its Do or Undo.
func recordStep(log *[]string, name string, doErr, undoErr error) Step {
	return Step{
		Name: name,
		Do: func() error {
			*log = append(*log, "do:"+name)
			return doErr
		},
		Undo: func() error {
			*log = append(*log, "undo:"+name)
			return undoErr
		},
	}
}

func TestTxnSuccessCallsNoUndo(t *testing.T) {
	var log []string
	var tx Txn
	tx.Add(recordStep(&log, "a", nil, nil))
	tx.Add(recordStep(&log, "b", nil, nil))
	tx.Add(recordStep(&log, "c", nil, nil))

	if err := tx.Run(); err != nil {
		t.Fatalf("Run() = %v, want nil", err)
	}
	want := []string{"do:a", "do:b", "do:c"}
	if !reflect.DeepEqual(log, want) {
		t.Fatalf("log = %v, want %v", log, want)
	}
}

func TestTxnFailureRollsBackCompletedInReverse(t *testing.T) {
	var log []string
	boom := errors.New("step c failed")
	var tx Txn
	tx.Add(recordStep(&log, "a", nil, nil))
	tx.Add(recordStep(&log, "b", nil, nil))
	tx.Add(recordStep(&log, "c", boom, nil))

	err := tx.Run()
	if err == nil {
		t.Fatalf("Run() = nil, want error")
	}
	if !errors.Is(err, boom) {
		t.Fatalf("Run() error %v does not wrap %v", err, boom)
	}
	// c's Do ran and failed (no undo for the failing step), then b and a
	// roll back in reverse.
	want := []string{"do:a", "do:b", "do:c", "undo:b", "undo:a"}
	if !reflect.DeepEqual(log, want) {
		t.Fatalf("log = %v, want %v", log, want)
	}
}

func TestTxnNilUndoSkippedOnRollback(t *testing.T) {
	var log []string
	boom := errors.New("step c failed")
	var tx Txn
	tx.Add(recordStep(&log, "a", nil, nil))
	tx.Add(Step{Name: "b-no-undo", Do: func() error { log = append(log, "do:b"); return nil }}) // nil Undo
	tx.Add(recordStep(&log, "c", boom, nil))

	if err := tx.Run(); err == nil {
		t.Fatalf("Run() = nil, want error")
	}
	// b has a nil Undo so it is skipped; only a rolls back.
	want := []string{"do:a", "do:b", "do:c", "undo:a"}
	if !reflect.DeepEqual(log, want) {
		t.Fatalf("log = %v, want %v", log, want)
	}
}

func TestTxnUndoErrorReportedButRollbackContinues(t *testing.T) {
	var log []string
	doErr := errors.New("step c failed")
	undoErr := errors.New("undo of b failed")
	var tx Txn
	tx.Add(recordStep(&log, "a", nil, nil))
	tx.Add(recordStep(&log, "b", nil, undoErr)) // its Undo errors
	tx.Add(recordStep(&log, "c", doErr, nil))

	err := tx.Run()
	if err == nil {
		t.Fatalf("Run() = nil, want error")
	}
	// Rollback continued past b's failing undo to a.
	want := []string{"do:a", "do:b", "do:c", "undo:b", "undo:a"}
	if !reflect.DeepEqual(log, want) {
		t.Fatalf("log = %v, want %v", log, want)
	}
	// The error must surface both the original cause and the undo failure.
	if !errors.Is(err, doErr) {
		t.Fatalf("error %v does not wrap original cause %v", err, doErr)
	}
	if !strings.Contains(err.Error(), "undo of b failed") {
		t.Fatalf("error %v does not mention the undo failure", err)
	}
}
