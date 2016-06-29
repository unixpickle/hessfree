package hessfree

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

const (
	linearizerTestOutputPrecision = 1e-5
	linearizerTestGradPrecision   = 1e-5
)

func TestLinearizerOutput(t *testing.T) {
	actual, expected, _ := linearizerTestOutputs()

	for i, x := range expected.Output() {
		a := actual.Output()[i]
		if math.Abs(a-x) > linearizerTestOutputPrecision {
			t.Error("output", i, "should be", x, "but it's", a)
		}
	}

	for i, x := range expected.ROutput() {
		a := actual.ROutput()[i]
		if math.Abs(a-x) > linearizerTestOutputPrecision {
			t.Error("r-output", i, "should be", x, "but it's", a)
		}
	}
}

func TestLinearizerGradient(t *testing.T) {
	rand.Seed(123)

	actual, expected, params := linearizerTestOutputs()

	upstream := make(linalg.Vector, len(actual.Output()))
	upstreamR := make(linalg.Vector, len(upstream))
	upstreamBackup := make(linalg.Vector, len(upstream))
	upstreamRBackup := make(linalg.Vector, len(upstream))
	for i := range upstream {
		upstream[i] = rand.NormFloat64()
		upstreamR[i] = rand.NormFloat64()
	}
	copy(upstreamBackup, upstream)
	copy(upstreamRBackup, upstreamR)

	grad := autofunc.NewGradient([]*autofunc.Variable{params})
	rgrad := autofunc.NewRGradient([]*autofunc.Variable{params})
	actual.PropagateRGradient(upstream, upstreamR, rgrad, grad)

	expectedGrad := autofunc.NewGradient([]*autofunc.Variable{params})
	expectedRGrad := autofunc.NewRGradient([]*autofunc.Variable{params})
	expected.PropagateRGradient(upstreamBackup, upstreamRBackup, expectedRGrad, expectedGrad)

	for variable, expectedVec := range expectedGrad {
		actualVec := grad[variable]
		for i, x := range expectedVec {
			a := actualVec[i]
			if math.Abs(a-x) > linearizerTestGradPrecision {
				t.Error("gradient entry", i, "should be", x, "but it's", a)
			}
		}
	}

	for variable, expectedVec := range expectedRGrad {
		actualVec := rgrad[variable]
		for i, x := range expectedVec {
			a := actualVec[i]
			if math.Abs(a-x) > linearizerTestGradPrecision {
				t.Error("r-gradient entry", i, "should be", x, "but it's", a)
			}
		}
	}
}

func linearizerTestOutputs() (actual, expected autofunc.RResult, params *autofunc.Variable) {
	params = &autofunc.Variable{Vector: []float64{0.78168, -0.26282}}
	lt := linearizerTest{XY: params}
	inputs := autofunc.NewRVariable(&autofunc.Variable{
		Vector: []float64{1, 2, -0.3, 0.3},
	}, autofunc.RVector{})
	deltaVar := &autofunc.Variable{Vector: []float64{-0.19416, 0.61623}}
	delta := ParamDelta{
		params: autofunc.NewRVariable(deltaVar,
			autofunc.RVector{deltaVar: []float64{0.333, -0.414}}),
	}
	linearizer := &Linearizer{Batcher: newLinearizerTestRBatcher(params)}

	expected = lt.LinearBatch(delta, inputs, len(inputs.Output())/2)
	actual = linearizer.LinearBatch(delta, inputs, len(inputs.Output())/2)
	return
}

type linearizerTest struct {
	XY *autofunc.Variable
}

func (l *linearizerTest) LinearBatch(d ParamDelta, ins autofunc.RResult, n int) autofunc.RResult {
	var results []autofunc.RResult
	for i := 0; i < n*2; i += 2 {
		a, b := ins.Output()[i], ins.Output()[i+1]
		mat := l.jacobian(a, b)
		initial := l.initialValue(a, b)
		jacobianProduct := mat.ApplyR(autofunc.RVector{}, d[l.XY])
		results = append(results, autofunc.AddR(jacobianProduct, initial))
	}
	return autofunc.ConcatR(results...)
}

func (l *linearizerTest) jacobian(a, b float64) *autofunc.LinTran {
	x := l.XY.Vector[0]
	y := l.XY.Vector[1]
	return &autofunc.LinTran{
		Rows: 2,
		Cols: 2,
		Data: &autofunc.Variable{
			Vector: []float64{
				math.Cos(a*x*x)*2*a*x - b*y*y*math.Sin(x*y),
				b*math.Cos(x*y) - b*y*x*math.Sin(x*y),
				y * math.Cos(x),
				math.Sin(x) - b*2*math.Cos(a*y)*math.Sin(a*y)*a,
			},
		},
	}
}

func (l *linearizerTest) initialValue(a, b float64) autofunc.RResult {
	x := l.XY.Vector[0]
	y := l.XY.Vector[1]
	f1 := math.Sin(a*x*x) + b*y*math.Cos(x*y)
	f2 := math.Sin(x)*y + b*math.Pow(math.Cos(a*y), 2)
	return autofunc.NewRVariable(&autofunc.Variable{
		Vector: []float64{f1, f2},
	}, autofunc.RVector{})
}

type linearizerTestFunc struct {
	XY *autofunc.Variable
}

func newLinearizerTestRBatcher(xy *autofunc.Variable) autofunc.RBatcher {
	return &autofunc.RFuncBatcher{F: &linearizerTestFunc{xy}}
}

func (l *linearizerTestFunc) Apply(in autofunc.Result) autofunc.Result {
	x := autofunc.Slice(l.XY, 0, 1)
	y := autofunc.Slice(l.XY, 1, 2)
	a := autofunc.Slice(in, 0, 1)
	b := autofunc.Slice(in, 1, 2)

	out1 := autofunc.Add(autofunc.Sin{}.Apply(autofunc.Mul(a, autofunc.Pow(x, 2))),
		autofunc.Mul(autofunc.Mul(b, y), autofunc.Cos{}.Apply(autofunc.Mul(x, y))))
	out2 := autofunc.Add(autofunc.Mul(y, autofunc.Sin{}.Apply(x)),
		autofunc.Mul(b, autofunc.Pow(autofunc.Cos{}.Apply(autofunc.Mul(a, y)), 2)))
	return autofunc.Concat(out1, out2)
}

func (l *linearizerTestFunc) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	x := autofunc.SliceR(autofunc.NewRVariable(l.XY, v), 0, 1)
	y := autofunc.SliceR(autofunc.NewRVariable(l.XY, v), 1, 2)
	a := autofunc.SliceR(in, 0, 1)
	b := autofunc.SliceR(in, 1, 2)

	out1 := autofunc.AddR(autofunc.Sin{}.ApplyR(v, autofunc.MulR(a, autofunc.PowR(x, 2))),
		autofunc.MulR(autofunc.MulR(b, y), autofunc.Cos{}.ApplyR(v, autofunc.MulR(x, y))))
	out2 := autofunc.AddR(autofunc.MulR(y, autofunc.Sin{}.ApplyR(v, x)),
		autofunc.MulR(b, autofunc.PowR(autofunc.Cos{}.ApplyR(v, autofunc.MulR(a, y)), 2)))
	return autofunc.ConcatR(out1, out2)
}
