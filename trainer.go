package hessfree

import "github.com/unixpickle/sgd"

const (
	defaultConvergenceMinK    = 10
	defaultConvergenceKScale  = 0.1
	defaultConvergenceEpsilon = 0.0005
)

// ConvergenceCriteria stores the parameters for the
// relative change convergence criteria described in
// Martens (2010).
// If the values are 0, defaults from Martens (2010)
// are used.
type ConvergenceCriteria struct {
	MinK    float64
	KScale  float64
	Epsilon float64
}

// A Trainer runs Hessian Free on a Learner.
type Trainer struct {
	// Learner is trained using Hessian Free.
	Learner Learner

	// Samples contains all of the training samples.
	Samples sgd.SampleSet

	// UI is the means by which the Trainer communicates with
	// the user, logging information and receiving termination
	// signals.
	UI UI

	// Convergence are the convergence criteria.
	Convergence ConvergenceCriteria
}

func (t *Trainer) Train() {
	// TODO: this.
	panic("not yet implemented.")
}
