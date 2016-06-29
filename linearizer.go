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

// LinearBatch applies the linearized function to a batch
// of inputs for the given parameter delta.
//
// The result supports back-propagation and R-propagation,
// but it cannot compute gradients or r-gradients for the
// inputs, which must be held constant.
func (l *Linearizer) LinearBatch(d ParamDelta, ins autofunc.RResult, n int) autofunc.RResult {
	output := l.Batcher.BatchR(d.outputRVector(), ins, n)
	outputR := l.Batcher.BatchR(d.rOutputRVector(), ins, n)
	return &linearizerRResult{
		OutputVec:     output.Output().Copy().Add(output.ROutput()),
		ROutputVec:    outputR.ROutput(),
		BatcherOutput: output,

		Inputs: ins,
		Delta:  d,
	}
}

type linearizerRResult struct {
	OutputVec     linalg.Vector
	ROutputVec    linalg.Vector
	BatcherOutput autofunc.RResult

	Inputs autofunc.RResult
	Delta  ParamDelta
}

func (l *linearizerRResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *linearizerRResult) ROutput() linalg.Vector {
	return l.ROutputVec
}

func (l *linearizerRResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	return l.BatcherOutput.Constant(rg, g)
}

func (l *linearizerRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rg autofunc.RGradient, g autofunc.Gradient) {
	if !l.Inputs.Constant(rg, g) {
		panic("linearized function's inputs must be constant")
	}
	if !l.BatcherOutput.Constant(rg, nil) {
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
}
