package hessfree

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// GaussNewtonNN approximates a neural network using
// its Gauss-Newton approximation, linearizing all
// the layers before the output layer and cost.
//
// This is a way to approximate a neural net as a
// convex function, provided that the cost function
// and output layer are matching as described in
// http://nic.schraudolph.org/pubs/Schraudolph02.pdf.
type GaussNewtonNN struct {
	// Layers is all of the layers of the neural net
	// up to the last one.
	Layers autofunc.RBatcher

	// Output is the output layer of the network.
	// If this is nil, the output from the Layers
	// is fed directly into the cost function.
	Output autofunc.RBatcher

	Cost neuralnet.CostFunc
}

// Quad evaluates the Gauss-Newton approximation
// at the given delta.
func (g *GaussNewtonNN) Quad(delta ConstParamDelta, s sgd.SampleSet) float64 {
	argDelta := ParamDelta{}
	for variable, d := range delta {
		argDelta[variable] = &autofunc.Variable{Vector: d}
	}
	return g.objective(argDelta, s).Output()[0]
}

// QuadGradient computes the gradient of Gauss-Newton
// approximation at the given delta.
func (g *GaussNewtonNN) QuadGrad(delta ConstParamDelta, s sgd.SampleSet) ConstParamDelta {
	argDelta := ParamDelta{}
	var tempVariables []*autofunc.Variable
	var mapVariables []*autofunc.Variable
	for variable, d := range delta {
		tempVar := &autofunc.Variable{Vector: d}
		argDelta[variable] = tempVar
		tempVariables = append(tempVariables, tempVar)
		mapVariables = append(mapVariables, variable)
	}
	output := g.objective(argDelta, s)

	grad := autofunc.NewGradient(tempVariables)
	output.PropagateGradient([]float64{1}, grad)

	res := ConstParamDelta{}
	for i, mapVariable := range mapVariables {
		res[mapVariable] = grad[tempVariables[i]]
	}
	return res
}

// QuadHessian applies the Hessian of the Gauss-Newton
// approximation to the given delta while simultaneously
// evaluating the approximation at x.
func (g *GaussNewtonNN) QuadHessian(delta, x ConstParamDelta, s sgd.SampleSet) (ConstParamDelta,
	float64) {
	rDelta := ParamRDelta{}
	var tempVariables []*autofunc.Variable
	var mapVariables []*autofunc.Variable
	for variable, d := range delta {
		tempVar := &autofunc.Variable{Vector: x[variable]}
		rDelta[variable] = &autofunc.RVariable{
			Variable:   tempVar,
			ROutputVec: d,
		}
		tempVariables = append(tempVariables, tempVar)
		mapVariables = append(mapVariables, variable)
	}
	output := g.objectiveR(rDelta, s)

	rgrad := autofunc.NewRGradient(tempVariables)
	output.PropagateRGradient([]float64{1}, []float64{0}, rgrad, nil)

	res := ConstParamDelta{}
	for i, mapVariable := range mapVariables {
		res[mapVariable] = rgrad[tempVariables[i]]
	}
	return res, output.Output()[0]
}

// ObjectiveAtZero applies the actual, unapproximated
// objective function to its underlying variables.
func (g *GaussNewtonNN) ObjectiveAtZero(s sgd.SampleSet) float64 {
	sampleIns, sampleOuts := joinSamples(s)
	inputs := &autofunc.Variable{Vector: sampleIns}
	output1 := g.Layers.Batch(inputs, s.Len())
	return g.outFunc(sampleOuts, s.Len()).Apply(output1).Output()[0]
}

// objective evaluates the approximated objective
// (cost) function given a SampleSet full of
// neuralnet.VectorSample instances.
//
// This will run all of the samples in one batch.
//
// The result can be back-propagated through to the
// parameter delta, but not through the parameters
// of the neural network's layers (as these are held
// constant while the layers are linearized).
func (g *GaussNewtonNN) objective(delta ParamDelta, s sgd.SampleSet) autofunc.Result {
	sampleIns, sampleOuts := joinSamples(s)
	layerOutput := LinApprox(g.Layers, delta, sampleIns, s.Len())
	x0 := layerOutput.(*linearizerResult).BatcherOutput.Output()
	return QuadApprox(g.outFunc(sampleOuts, s.Len()), x0, layerOutput)
}

// objectiveR is like objective, but for RResults.
func (g *GaussNewtonNN) objectiveR(delta ParamRDelta, s sgd.SampleSet) autofunc.RResult {
	sampleIns, sampleOuts := joinSamples(s)
	layerOutput := LinApproxR(g.Layers, delta, sampleIns, s.Len())
	x0 := layerOutput.(*linearizerRResult).BatcherOutput.Output()
	return QuadApproxR(g.outFunc(sampleOuts, s.Len()), x0, layerOutput)
}

func (g *GaussNewtonNN) outFunc(expectedOuts linalg.Vector, n int) autofunc.RFunc {
	return &netOutFunc{
		LastLayer:   g.Output,
		CostFunc:    g.Cost,
		SampleOuts:  expectedOuts,
		SampleCount: n,
	}
}

func joinSamples(s sgd.SampleSet) (ins, outs linalg.Vector) {
	if s.Len() == 0 {
		return
	}

	sample := s.GetSample(0).(neuralnet.VectorSample)
	ins = make(linalg.Vector, len(sample.Input)*s.Len())
	outs = make(linalg.Vector, len(sample.Output)*s.Len())
	copy(ins, sample.Input)
	copy(outs, sample.Output)

	for i := 1; i < s.Len(); i++ {
		sample = s.GetSample(i).(neuralnet.VectorSample)
		copy(ins[i*len(sample.Input):(i+1)*len(sample.Input)], sample.Input)
		copy(outs[i*len(sample.Output):(i+1)*len(sample.Output)], sample.Output)
	}

	return
}

type netOutFunc struct {
	LastLayer   autofunc.RBatcher
	CostFunc    neuralnet.CostFunc
	SampleOuts  linalg.Vector
	SampleCount int
}

func (n *netOutFunc) Apply(in autofunc.Result) autofunc.Result {
	if n.LastLayer != nil {
		out1 := n.LastLayer.Batch(in, n.SampleCount)
		return n.CostFunc.Cost(n.SampleOuts, out1)
	} else {
		return n.CostFunc.Cost(n.SampleOuts, in)
	}
}

func (n *netOutFunc) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	if n.LastLayer != nil {
		out1 := n.LastLayer.BatchR(v, in, n.SampleCount)
		return n.CostFunc.CostR(v, n.SampleOuts, out1)
	} else {
		return n.CostFunc.CostR(v, n.SampleOuts, in)
	}
}
