package hessfree

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A Linearizer approxmitase an autofunc.RBatcher as a
// linear function of its underlying variables.
// The batcher is not linearized with respect to its
// actual inputs, as these are generally constant while
// training/optimization (for a mini-batch).
//
// For a neural network, part of approximating the
// Gauss-Newton matrix involves linearizing all of the
// layers up to the output and cost layers.
// This can be done by wrapping said layers in a single
// Linearizer.
type Linearizer struct {
	Batcher autofunc.RBatcher
}

// LinearBatch evaluates the linearized function on a
// parameter delta and a batch of (constant) inputs.
//
// The result supports back-propagation and R-propagation
// through the parameter delta.
func (l *Linearizer) LinearBatch(d ParamDelta, ins linalg.Vector, n int) autofunc.RResult {
	insVar := &autofunc.Variable{Vector: ins}
	insRVar := autofunc.NewRVariable(insVar, autofunc.RVector{})

	output := l.Batcher.BatchR(d.outputRVector(), insRVar, n)
	outputR := l.Batcher.BatchR(d.rOutputRVector(), insRVar, n)
	return &linearizerRResult{
		OutputVec:     output.Output().Copy().Add(output.ROutput()),
		ROutputVec:    outputR.ROutput(),
		BatcherOutput: output,

		Delta: d,
	}
}

type linearizerRResult struct {
	OutputVec     linalg.Vector
	ROutputVec    linalg.Vector
	BatcherOutput autofunc.RResult

	Delta ParamDelta
}

func (l *linearizerRResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *linearizerRResult) ROutput() linalg.Vector {
	return l.ROutputVec
}

func (l *linearizerRResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	for _, r := range l.Delta {
		if !r.Constant(rg, g) {
			return false
		}
	}
	return true
}

func (l *linearizerRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rg autofunc.RGradient, g autofunc.Gradient) {
	gradient := l.Delta.zeroGradient()
	rGradient := l.Delta.zeroGradient()

	// Back-propagation is equivalent to left-multiplication by the Jacobian.
	zeroVec := make(linalg.Vector, len(upstream))
	l.BatcherOutput.PropagateRGradient(upstream, zeroVec, autofunc.RGradient{},
		gradient)
	l.BatcherOutput.PropagateRGradient(upstreamR, zeroVec, autofunc.RGradient{},
		rGradient)

	for variable, downstream := range gradient {
		downstreamR := rGradient[variable]
		l.Delta[variable].PropagateRGradient(downstream, downstreamR, rg, g)
	}
}
