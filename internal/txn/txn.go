// Package txn is a tiny transaction runner: an ordered list of steps, each with
// a forward action (Do) and a compensating action (Undo). If any Do fails, the
// runner invokes Undo for every already-completed step in reverse order, so a
// mutating command is atomic and fails closed — nothing is left half-applied.
package txn

import (
	"errors"
	"fmt"
)

// Step is one unit of a transaction. Do performs the action; Undo reverses it
// during rollback. A nil Undo means the step needs no compensation and is
// skipped on rollback (e.g. a pure verify).
type Step struct {
	Name string
	Do   func() error
	Undo func() error
}

// Txn is an ordered set of steps run as an all-or-nothing transaction.
type Txn struct {
	steps []Step
}

// Add appends a step. Steps run in the order added.
func (t *Txn) Add(s Step) {
	t.steps = append(t.steps, s)
}

// Run executes each step's Do in order. On the first Do that errors, it rolls
// back: it invokes Undo for every already-completed step in reverse order
// (skipping nil Undos), then returns an error wrapping the original failure.
// Undo failures do not stop the rollback; they are collected and appended to
// the returned error so they are surfaced, not swallowed. A fully successful
// transaction returns nil and calls no Undo.
func (t *Txn) Run() error {
	completed := make([]Step, 0, len(t.steps))
	for _, s := range t.steps {
		if err := s.Do(); err != nil {
			undoErrs := rollback(completed)
			cause := fmt.Errorf("step %q failed: %w", s.Name, err)
			if len(undoErrs) > 0 {
				return fmt.Errorf("%w; rollback errors: %w", cause, errors.Join(undoErrs...))
			}
			return cause
		}
		completed = append(completed, s)
	}
	return nil
}

// rollback runs the Undo of each completed step in reverse order, skipping nil
// Undos, and returns any undo errors (rollback continues past a failing undo).
func rollback(completed []Step) []error {
	var errs []error
	for i := len(completed) - 1; i >= 0; i-- {
		s := completed[i]
		if s.Undo == nil {
			continue
		}
		if err := s.Undo(); err != nil {
			errs = append(errs, fmt.Errorf("undo %q: %w", s.Name, err))
		}
	}
	return errs
}
