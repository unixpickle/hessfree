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
	Layers neuralnet.Network
	Output neuralnet.Network
	Cost   neuralnet.CostFunc
}

// Objective evaluates the approximated objective
// (cost) function given a SampleSet full of
// neuralnet.VectorSample instances.
//
// This will run all of the samples in one batch.
//
// The result can be back-propagated through to the
// parameter delta, but not through the parameters
// of the neural network's layers (as these are held
// constant while the layers are linearized).
func (g *GaussNewtonNN) Objective(delta ParamDelta, s sgd.SampleSet) autofunc.Result {
	sampleIns, sampleOuts := joinSamples(s)

	linearizer := Linearizer{Batcher: g.Layers.BatchLearner()}
	layerOutput := linearizer.LinearBatch(delta, sampleIns, s.Len())
	netOutput := g.Output.BatchLearner().Batch(layerOutput, s.Len())

	return g.Cost.Cost(sampleOuts, netOutput)
}

// ObjectiveR is like Objective, but for RResults.
func (g *GaussNewtonNN) ObjectiveR(delta ParamRDelta, s sgd.SampleSet) autofunc.RResult {
	sampleIns, sampleOuts := joinSamples(s)

	linearizer := Linearizer{Batcher: g.Layers.BatchLearner()}
	layerOutput := linearizer.LinearBatchR(delta, sampleIns, s.Len())
	netOutput := g.Output.BatchLearner().BatchR(autofunc.RVector{}, layerOutput, s.Len())

	return g.Cost.CostR(autofunc.RVector{}, sampleOuts, netOutput)
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
