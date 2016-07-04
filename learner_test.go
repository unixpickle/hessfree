package hessfree

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

const (
	learnerTestPrec   = 1e-5
	learnerTestOffset = 1e-1
)

func TestDampedNeuralLearner(t *testing.T) {
	network := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  30,
			OutputCount: 20,
		},
		neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  20,
			OutputCount: 30,
		},
	}
	network.Randomize()

	learner := &DampingLearner{
		WrappedLearner: &NeuralNetLearner{
			Layers:         network,
			Output:         nil,
			Cost:           neuralnet.SigmoidCECost{},
			MaxSubBatch:    10,
			MaxConcurrency: 1,
		},
		DampingCoeff: 1,
	}

	var inputs []linalg.Vector
	for i := 0; i < 50; i++ {
		vec := make(linalg.Vector, 30)
		for i := range vec {
			vec[i] = rand.Float64()
		}
		inputs = append(inputs, vec)
	}
	sampleSet := neuralnet.VectorSampleSet(inputs, inputs)

	testLearner(t, learner, sampleSet)
}

func BenchmarkDampedNeuralLearnerQuadHessian(b *testing.B) {
	network := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  100,
			OutputCount: 1000,
		},
		neuralnet.Sigmoid{},
		&neuralnet.DenseLayer{
			InputCount:  1000,
			OutputCount: 100,
		},
	}
	network.Randomize()

	learner := &DampingLearner{
		WrappedLearner: &NeuralNetLearner{
			Layers:         network,
			Output:         nil,
			Cost:           neuralnet.SigmoidCECost{},
			MaxSubBatch:    10,
			MaxConcurrency: 1,
		},
		DampingCoeff: 1,
	}

	var inputs []linalg.Vector
	for i := 0; i < 50; i++ {
		vec := make(linalg.Vector, 100)
		for i := range vec {
			vec[i] = rand.Float64()
		}
		inputs = append(inputs, vec)
	}
	sampleSet := neuralnet.VectorSampleSet(inputs, inputs)

	benchLearnerQuadApprox(b, learner, sampleSet)
}

func testLearner(t *testing.T, l Learner, s sgd.SampleSet) {
	objective := l.MakeObjective()

	destination := ConstParamDelta{}
	zeroDelta := ConstParamDelta{}
	for _, v := range l.Parameters() {
		zeroDelta[v] = make(linalg.Vector, len(v.Vector))
		vec := make(linalg.Vector, len(v.Vector))
		destination[v] = vec
		for i := range vec {
			vec[i] = rand.NormFloat64() * learnerTestOffset
		}
	}

	value := objective.Quad(zeroDelta, s)
	grad := objective.QuadGrad(zeroDelta, s)
	rgrad, valueFromRGrad := objective.QuadHessian(destination, zeroDelta, s)

	if math.Abs(value-valueFromRGrad) > learnerTestPrec {
		t.Error("Quad() gave", value, "but QuadHessian() gave", valueFromRGrad)
	}

	expectedVal := value + grad.dot(destination) + 0.5*rgrad.dot(destination)
	actualVal := objective.Quad(destination, s)
	if math.Abs(actualVal-expectedVal) > learnerTestPrec {
		t.Error("expected output", expectedVal, "got output", actualVal)
	}

	grad.addDelta(rgrad, 1)
	actualGrad := objective.QuadGrad(destination, s)

GradLoop:
	for variable, expGrad := range grad {
		actGrad := actualGrad[variable]
		for i, x := range expGrad {
			a := actGrad[i]
			if math.Abs(a-x) > learnerTestPrec {
				t.Error("expected grad value", x, "but got", a)
				break GradLoop
			}
		}
	}
}

func benchLearnerQuadApprox(b *testing.B, l Learner, s sgd.SampleSet) {
	destination := ConstParamDelta{}
	for _, v := range l.Parameters() {
		vec := make(linalg.Vector, len(v.Vector))
		destination[v] = vec
		for i := range vec {
			vec[i] = rand.NormFloat64() * learnerTestOffset
		}
	}

	obj := l.MakeObjective()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		obj.QuadHessian(destination, destination, s)
	}
}
