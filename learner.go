package hessfree

import (
	"fmt"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

const defaultDampingCoeff = 1

// A Learner has learnable parameters and can create
// Objectives based on a sample set and the current
// set of parameters.
type Learner interface {
	// Parameters returns the learnable parameters that may
	// be adjusted by Adjust().
	Parameters() []*autofunc.Variable

	// MakeObjective creates an objective whose approximation
	// is centered around the current underlying variables.
	//
	// This should be called once before every Adjust() call,
	// yielding a cycle of MakeObjective and Adjust.
	// This way, each Objective can be tweaked based on info
	// about the previous cycle (e.g. for damping).
	MakeObjective() Objective

	// Adjust updates the parameters after an Objective
	// (created by MakeObjective()) has been minimized.
	//
	// Of the delta parameters, adjustment is the recommended
	// change to the underlying parameters (bactracked), whereas
	// quadMin is the last iterate of Conjugate Gradients,
	// reflecting the supposed minimum of the quadratic objective.
	//
	// The provided SampleSet is the set of samples for which
	// the given delta supposedly to improves the cost.
	// Adjust may need this sample set to analyze the effects
	// of the delta, e.g. for damping purposes.
	Adjust(adjustment, quadMin ConstParamDelta, s sgd.SampleSet)
}

// A NeuralNetLearner is a Learner which wraps a neural net
// and creates concurrent Gauss-Newton objectives.
type NeuralNetLearner struct {
	// Parameters for the GaussNewtonNN objectives.
	Layers neuralnet.Network
	Output neuralnet.Network
	Cost   neuralnet.CostFunc

	// Parameters for the ConcurrentObjectives.
	MaxSubBatch    int
	MaxConcurrency int
}

// Parameters returns the parameters of n.Layers.
func (n *NeuralNetLearner) Parameters() []*autofunc.Variable {
	return n.Layers.Parameters()
}

// MakeObjective creates a ConcurrentObjective which
// wraps a Gauss-Newton objective.
func (n *NeuralNetLearner) MakeObjective() Objective {
	var output autofunc.RBatcher
	if n.Output != nil {
		output = n.Output.BatchLearner()
	}
	return &ConcurrentObjective{
		Wrapped: &GaussNewtonNN{
			Layers: n.Layers.BatchLearner(),
			Output: output,
			Cost:   n.Cost,
		},
		MaxConcurrency: n.MaxConcurrency,
		MaxSubBatch:    n.MaxSubBatch,
	}
}

// Adjust adds the delta to its parameters.
func (n *NeuralNetLearner) Adjust(d, m ConstParamDelta, s sgd.SampleSet) {
	d.addToVars()
}

// A DampingLearner wraps a learner in the damping
// mechanism described in Martens (2010).
type DampingLearner struct {
	WrappedLearner Learner

	// DampingCoeff is the coefficient for the squared
	// deltas in the damping term.
	// It is adjusted during training using the heuristic
	// described in Martens (2010).
	// If DampingCoeff is 0, it will be set to a default
	// value during the first training iteration.
	//
	// During damping, this coefficient is multiplied by
	// the number of samples in each sample set, since it
	// is assumed that the total cost is the sum of the
	// costs for each sample.
	DampingCoeff float64

	// UseQuadMin can be used to specify that the minimum of
	// the quadratic should be used to adjust damping, as
	// opposed to the backtracked value.
	UseQuadMin bool

	// If UI is set, it will be used to log damping updates.
	UI UI

	lastObjective Objective
}

func (d *DampingLearner) Parameters() []*autofunc.Variable {
	return d.WrappedLearner.Parameters()
}

func (d *DampingLearner) MakeObjective() Objective {
	if d.DampingCoeff == 0 {
		d.DampingCoeff = defaultDampingCoeff
	}
	d.lastObjective = d.WrappedLearner.MakeObjective()
	return &dampedObjective{
		WrappedObjective: d.lastObjective,
		Coeff:            d.DampingCoeff,
	}
}

func (d *DampingLearner) Adjust(delta, quadMin ConstParamDelta, s sgd.SampleSet) {
	var trust float64

	centerVal := d.lastObjective.Objective(ConstParamDelta{}, s)
	if d.UseQuadMin {
		quadOffset := d.lastObjective.Quad(quadMin, s)
		realOffset := d.lastObjective.Objective(quadMin, s)
		trust = (realOffset - centerVal) / (quadOffset - centerVal)
	} else {
		quadOffset := d.lastObjective.Quad(delta, s)
		realOffset := d.lastObjective.Objective(delta, s)
		trust = (realOffset - centerVal) / (quadOffset - centerVal)
	}

	d.WrappedLearner.Adjust(delta, quadMin, s)

	d.UI.Log("DampingLearner", fmt.Sprintf("trust quotient is %f", trust))
	if trust < 0.25 {
		d.DampingCoeff *= 3.0 / 2.0
		if d.UI != nil {
			d.UI.Log("DampingLearner", fmt.Sprintf("raised damping to %f",
				d.DampingCoeff))
		}
	} else if trust > 0.75 {
		d.DampingCoeff *= 2.0 / 3.0
		if d.UI != nil {
			d.UI.Log("DampingLearner", fmt.Sprintf("lowered damping to %f",
				d.DampingCoeff))
		}
	}
}

type dampedObjective struct {
	WrappedObjective Objective
	Coeff            float64
}

func (d *dampedObjective) Quad(delta ConstParamDelta, s sgd.SampleSet) float64 {
	res := d.WrappedObjective.Quad(delta, s)
	scaler := float64(s.Len()) * d.Coeff
	for _, subDelta := range delta {
		for _, x := range subDelta {
			res += scaler * x * x
		}
	}
	return res
}

func (d *dampedObjective) QuadGrad(delta ConstParamDelta, s sgd.SampleSet) ConstParamDelta {
	res := d.WrappedObjective.QuadGrad(delta, s)

	scaler := float64(2*s.Len()) * d.Coeff
	for variable, subDelta := range delta {
		resVec := res[variable]
		for i, x := range subDelta {
			resVec[i] += scaler * x
		}
	}

	return res
}

func (d *dampedObjective) QuadHessian(delta, x ConstParamDelta, s sgd.SampleSet) (ConstParamDelta,
	float64) {
	res, outVal := d.WrappedObjective.QuadHessian(delta, x, s)

	scaler := float64(s.Len()) * d.Coeff
	rScaler := 2 * scaler
	for variable, subDelta := range delta {
		resVec := res[variable]
		for i, x := range subDelta {
			resVec[i] += rScaler * x
		}
	}
	for _, subDelta := range x {
		for _, y := range subDelta {
			outVal += scaler * y * y
		}
	}

	return res, outVal
}

func (d *dampedObjective) Objective(delta ConstParamDelta, s sgd.SampleSet) float64 {
	return d.WrappedObjective.Objective(delta, s)
}
