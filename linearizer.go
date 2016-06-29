package hessfree

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A ParamDelta is a measure (t - t0) where t is a set of
// parameters and t0 is an initial set of parameters.
type ParamDelta autofunc.RVector

// A Linearizer approxmitase an autofunc.RBatcher as a
// linear function of its underlying parameters.
// The underlying function is not linearized with respect
// to its actual inputs, as these are generally constant
// during training/optimization (for a mini-batch).
//
// For a neural network, part of approximating the
// Gauss-Newton matrix involves linearizing all of the
// layers up to the output and cost layers.
// This can be done by wrapping said layers in a single
// Linearizer.
type Linearizer struct {
	Batcher autofunc.RBatcher
}

// BatchR applies the linearized function to a batch of
// inputs, using the given parameter delta and its
// corresponding rVec.
func (l *Linearizer) BatchR(rVec autofunc.RVector, d ParamDelta, ins autofunc.RResult,
	n int) autofunc.RResult {
	outputR := l.Batcher.BatchR(rVec, ins, n)
	output := l.Batcher.BatchR(autofunc.RVector(d), ins, n)
	return &linearizerRResult{
		OutputVec:  output.Output().Copy().Add(output.ROutput()),
		ROutputVec: outputR.ROutput(),
	}
}

type linearizerRResult struct {
	OutputVec  linalg.Vector
	ROutputVec linalg.Vector

	BatcherOutput  autofunc.RResult
	BatcherROutput autofunc.RResult
}

func (l *linearizerRResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *linearizerRResult) ROutput() linalg.Vector {
	return l.ROutputVec
}

func (l *linearizerRResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	return l.BatcherOutput.Constant(rg, g) && l.BatcherROutput.Constant(rg, g)
}

func (l *linearizerRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rg autofunc.RGradient, g autofunc.Gradient) {
	zeroVec := make(linalg.Vector, len(upstream))
	l.BatcherROutput.PropagateRGradient(upstream, zeroVec, autofunc.RGradient{}, g)
	l.BatcherROutput.PropagateRGradient(upstreamR, zeroVec, autofunc.RGradient{},
		autofunc.Gradient(rg))
}
