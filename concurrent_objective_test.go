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
	objectiveTestPrec       = 1e-5
	objectiveTestInSize     = 5
	objectiveTestHiddenSize = 2
	objectiveTestOutputSize = 3
	objectiveTestDeltaMag   = 0.4
)

func TestConcurrentObjectiveBasic(t *testing.T) {
	obj, delta := objectiveTestFunc()
	samples := objectiveTestSamples(1)

	concurrentObj := &ConcurrentObjective{
		MaxConcurrency: 1,
		MaxSubBatch:    1,
		Wrapped:        obj,
	}

	testObjectiveEquivalence(t, concurrentObj, obj, delta, samples)
}

func objectiveTestFunc() (*GaussNewtonNN, ConstParamDelta) {
	rand.Seed(123)
	net := &neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  objectiveTestInSize,
			OutputCount: objectiveTestHiddenSize,
		},
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  objectiveTestHiddenSize,
			OutputCount: objectiveTestOutputSize,
		},
	}
	net.Randomize()
	outputLayer := &neuralnet.Network{&neuralnet.LogSoftmaxLayer{}}

	gn := &GaussNewtonNN{
		Layers: net.BatchLearner(),
		Output: outputLayer.BatchLearner(),
		Cost:   neuralnet.DotCost{},
	}

	delta := ConstParamDelta{}
	for _, variable := range net.Parameters() {
		delta[variable] = make(linalg.Vector, len(variable.Vector))
		for i := range delta[variable] {
			delta[variable][i] = rand.NormFloat64() * objectiveTestDeltaMag
		}
	}

	return gn, delta
}

func objectiveTestSamples(count int) sgd.SampleSet {
	var samples sgd.SliceSampleSet
	for i := 0; i < count; i++ {
		vecSample := neuralnet.VectorSample{
			Input:  make(linalg.Vector, objectiveTestInSize),
			Output: make(linalg.Vector, objectiveTestOutputSize),
		}
		vecSample.Output[rand.Intn(objectiveTestOutputSize)] = 1
		for i := range vecSample.Input {
			vecSample.Input[i] = rand.NormFloat64()
		}
		samples = append(samples, vecSample)
	}
	return samples
}

func testObjectiveEquivalence(t *testing.T, actual, expected Objective, delta ConstParamDelta,
	s sgd.SampleSet) {
	actualOut := actual.Objective(delta, s)
	expectedOut := expected.Objective(delta, s)
	if math.Abs(actualOut-expectedOut) > objectiveTestPrec {
		t.Error("output should be", expectedOut, "but got", actualOut)
	}

	actualGrad := actual.ObjectiveGrad(delta, s)
	expectedGrad := expected.ObjectiveGrad(delta, s)
GradCheckLoop:
	for variable, actualVec := range actualGrad {
		expectedVec := expectedGrad[variable]
		for i, a := range actualVec {
			x := expectedVec[i]
			if math.Abs(a-x) > objectiveTestPrec {
				t.Error("partial", i, "should be", x, "but got", a)
				break GradCheckLoop
			}
		}
	}

	actualHess := actual.ObjectiveHessian(delta, s)
	expectedHess := expected.ObjectiveHessian(delta, s)
HessCheckLoop:
	for variable, actualVec := range actualHess {
		expectedVec := expectedHess[variable]
		for i, a := range actualVec {
			x := expectedVec[i]
			if math.Abs(a-x) > objectiveTestPrec {
				t.Error("hessian product", i, "should be", x, "but got", a)
				break HessCheckLoop
			}
		}
	}
}
