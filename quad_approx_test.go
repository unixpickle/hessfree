package hessfree

import (
	"math"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

const (
	quadApproxTestPrec = 1e-5
)

func TestQuadApproxOutput(t *testing.T) {
	inputVec := &autofunc.Variable{Vector: []float64{-0.383238, 0.945592}}
	centerVec := linalg.Vector{0.77892, 0.57992}

	expected := quadApproxTestEval(centerVec, inputVec)
	actual := QuadApprox(quadApproxTestFunc{}, centerVec, inputVec)

	if math.Abs(expected.Output()[0]-actual.Output()[0]) > quadApproxTestPrec {
		t.Error("expected output", expected.Output(), "but got", actual.Output()[0])
	}
}

func TestQuadApproxROutput(t *testing.T) {
	inputVec := &autofunc.Variable{Vector: []float64{-0.383238, 0.945592}}
	inputRVec := &autofunc.RVariable{
		Variable:   inputVec,
		ROutputVec: linalg.Vector{0.23427, -0.57973},
	}
	centerVec := linalg.Vector{0.77892, 0.57992}

	expected := quadApproxTestEvalR(centerVec, inputRVec)
	actual := QuadApproxR(quadApproxTestFunc{}, centerVec, inputRVec)

	if math.Abs(expected.Output()[0]-actual.Output()[0]) > quadApproxTestPrec {
		t.Error("expected output", expected.Output(), "but got", actual.Output()[0])
	}
	if math.Abs(expected.ROutput()[0]-actual.ROutput()[0]) > quadApproxTestPrec {
		t.Error("expected r-output", expected.ROutput(), "but got", actual.ROutput()[0])
	}
}

func TestQuadApproxGradient(t *testing.T) {
	inputVec := &autofunc.Variable{Vector: []float64{-0.383238, 0.945592}}
	inputRVec := &autofunc.RVariable{
		Variable:   inputVec,
		ROutputVec: linalg.Vector{0.23427, -0.57973},
	}
	centerVec := linalg.Vector{0.77892, 0.57992}

	expected := quadApproxTestEvalR(centerVec, inputRVec)
	actual := QuadApproxR(quadApproxTestFunc{}, centerVec, inputRVec)

	upstream1 := []float64{0.61045}
	upstreamR1 := []float64{-0.31045}
	upstream2 := []float64{0.61045}
	upstreamR2 := []float64{-0.31045}

	expectedGrad := autofunc.NewGradient([]*autofunc.Variable{inputVec})
	expectedRGrad := autofunc.NewRGradient([]*autofunc.Variable{inputVec})
	expected.PropagateRGradient(upstream1, upstreamR1, expectedRGrad, expectedGrad)

	actualGrad := autofunc.NewGradient([]*autofunc.Variable{inputVec})
	actualRGrad := autofunc.NewRGradient([]*autofunc.Variable{inputVec})
	actual.PropagateRGradient(upstream2, upstreamR2, actualRGrad, actualGrad)

	for variable, xVec := range expectedGrad {
		aVec := actualGrad[variable]
		for i, x := range xVec {
			a := aVec[i]
			if math.Abs(x-a) > quadApproxTestPrec {
				t.Error("expected partial", i, "to be", x, "but it's", a)
			}
		}
	}

	for variable, xVec := range expectedRGrad {
		aVec := actualRGrad[variable]
		for i, x := range xVec {
			a := aVec[i]
			if math.Abs(x-a) > quadApproxTestPrec {
				t.Error("expected r-partial", i, "to be", x, "but it's", a)
			}
		}
	}
}

func TestQuadApproxRGradient(t *testing.T) {
	inputVec := &autofunc.Variable{Vector: []float64{-0.383238, 0.945592}}
	centerVec := linalg.Vector{0.77892, 0.57992}

	expected := quadApproxTestEval(centerVec, inputVec)
	actual := QuadApprox(quadApproxTestFunc{}, centerVec, inputVec)

	upstream1 := []float64{0.61045}
	upstream2 := []float64{0.61045}

	expectedGrad := autofunc.NewGradient([]*autofunc.Variable{inputVec})
	expected.PropagateGradient(upstream1, expectedGrad)

	actualGrad := autofunc.NewGradient([]*autofunc.Variable{inputVec})
	actual.PropagateGradient(upstream2, actualGrad)

	for variable, xVec := range expectedGrad {
		aVec := actualGrad[variable]
		for i, x := range xVec {
			a := aVec[i]
			if math.Abs(x-a) > quadApproxTestPrec {
				t.Error("expected partial", i, "to be", x, "but it's", a)
			}
		}
	}
}

type quadApproxTestFunc struct{}

func (_ quadApproxTestFunc) Apply(in autofunc.Result) autofunc.Result {
	x := autofunc.Slice(in, 0, 1)
	y := autofunc.Slice(in, 1, 2)
	return autofunc.Add(autofunc.Sin{}.Apply(autofunc.Pow(x, 2)),
		autofunc.Mul(y, autofunc.Cos{}.Apply(autofunc.Mul(x, y))))
}

func (_ quadApproxTestFunc) ApplyR(v autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	x := autofunc.SliceR(in, 0, 1)
	y := autofunc.SliceR(in, 1, 2)
	return autofunc.AddR(autofunc.Sin{}.ApplyR(v, autofunc.PowR(x, 2)),
		autofunc.MulR(y, autofunc.Cos{}.ApplyR(v, autofunc.MulR(x, y))))
}

func quadApproxTestEval(center linalg.Vector, args autofunc.Result) autofunc.Result {
	bias, grad, hess := quadApproxTestTerms(center)

	centerVar := &autofunc.Variable{Vector: center.Copy().Scale(-1)}
	offset := autofunc.Add(args, centerVar)

	gradDot := autofunc.SumAll(autofunc.Mul(grad, offset))
	hessProduct := hess.Apply(offset)
	hessDot := autofunc.Scale(autofunc.SumAll(autofunc.Mul(hessProduct, offset)), 0.5)
	return autofunc.Add(autofunc.Add(bias, gradDot), hessDot)
}

func quadApproxTestEvalR(center linalg.Vector, args autofunc.RResult) autofunc.RResult {
	bias, grad, hess := quadApproxTestTermsR(center)

	centerVar := &autofunc.Variable{Vector: center.Copy().Scale(-1)}
	centerRVar := autofunc.NewRVariable(centerVar, autofunc.RVector{})
	offset := autofunc.AddR(args, centerRVar)

	gradDot := autofunc.SumAllR(autofunc.MulR(grad, offset))
	hessProduct := hess.ApplyR(autofunc.RVector{}, offset)
	hessDot := autofunc.ScaleR(autofunc.SumAllR(autofunc.MulR(hessProduct, offset)), 0.5)
	return autofunc.AddR(autofunc.AddR(bias, gradDot), hessDot)
}

func quadApproxTestTerms(center linalg.Vector) (bias, grad *autofunc.Variable,
	hessian *autofunc.LinTran) {
	x, y := center[0], center[1]

	value := math.Sin(x*x) + y*math.Cos(x*y)

	gradient := linalg.Vector{
		math.Cos(x*x)*2*x - y*y*math.Sin(x*y),
		math.Cos(x*y) - math.Sin(x*y)*x*y,
	}

	dx2 := -4*x*x*math.Sin(x*x) + 2*math.Cos(x*x) + y*y*y*(-math.Cos(x*y))
	dy2 := -x * (2*math.Sin(x*y) + x*y*math.Cos(x*y))
	dxy := -y * (2*math.Sin(x*y) + x*y*math.Cos(x*y))

	bias = &autofunc.Variable{Vector: []float64{value}}
	grad = &autofunc.Variable{Vector: gradient}
	hessian = &autofunc.LinTran{
		Rows: 2,
		Cols: 2,
		Data: &autofunc.Variable{Vector: []float64{dx2, dxy, dxy, dy2}},
	}
	return
}

func quadApproxTestTermsR(center linalg.Vector) (bias, grad *autofunc.RVariable,
	hessian *autofunc.LinTran) {
	biasVar, gradVar, hessian := quadApproxTestTerms(center)
	bias = autofunc.NewRVariable(biasVar, autofunc.RVector{})
	grad = autofunc.NewRVariable(gradVar, autofunc.RVector{})
	return
}
