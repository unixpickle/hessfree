package hessfree

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// LinApprox approximates the batcher as a linear
// function of its underlying variables, holding its
// inputs constant.
// The linear approximation is centered around the
// batcher's current underlying variables, and the
// argument to the approximation's Jacobian is given
// as a ParamDelta.
//
// The result supports back-propagation through the
// parameter delta.
func LinApprox(b autofunc.RBatcher, d ParamDelta, ins linalg.Vector, n int) autofunc.Result {
	insVar := &autofunc.Variable{Vector: ins}
	insRVar := autofunc.NewRVariable(insVar, autofunc.RVector{})
	output := b.BatchR(d.outputRVector(), insRVar, n)
	return &linearizerResult{
		OutputVec:     output.Output().Copy().Add(output.ROutput()),
		BatcherOutput: output,
		Delta:         d,
	}
}

// LinApproxR is like LinApprox but with R-operator
// support.
func LinApproxR(b autofunc.RBatcher, d ParamRDelta, ins linalg.Vector, n int) autofunc.RResult {
	insVar := &autofunc.Variable{Vector: ins}
	insRVar := autofunc.NewRVariable(insVar, autofunc.RVector{})

	output := b.BatchR(d.outputRVector(), insRVar, n)
	outputR := b.BatchR(d.rOutputRVector(), insRVar, n)
	return &linearizerRResult{
		OutputVec:     output.Output().Copy().Add(output.ROutput()),
		ROutputVec:    outputR.ROutput(),
		BatcherOutput: output,

		Delta: d,
	}
}

type linearizerResult struct {
	OutputVec     linalg.Vector
	BatcherOutput autofunc.RResult
	Delta         ParamDelta
}

func (l *linearizerResult) Output() linalg.Vector {
	return l.OutputVec
}

func (l *linearizerResult) Constant(g autofunc.Gradient) bool {
	for _, r := range l.Delta {
		if !r.Constant(g) {
			return false
		}
	}
	return true
}

func (l *linearizerResult) PropagateGradient(upstream linalg.Vector, g autofunc.Gradient) {
	gradient := l.Delta.zeroGradient()

	// TODO: optimize this if Delta is full of *autofunc.Variables.

	// Back-propagation is equivalent to left-multiplication by the Jacobian.
	zeroVec := make(linalg.Vector, len(upstream))
	l.BatcherOutput.PropagateRGradient(upstream, zeroVec, autofunc.RGradient{}, gradient)

	for variable, downstream := range gradient {
		l.Delta[variable].PropagateGradient(downstream, g)
	}
}

type linearizerRResult struct {
	OutputVec     linalg.Vector
	ROutputVec    linalg.Vector
	BatcherOutput autofunc.RResult

	Delta ParamRDelta
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

	// TODO: optimize this if Delta is full of *autofunc.RVariables.

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
