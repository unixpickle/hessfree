package hessfree

import (
	"sync"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// QuadApprox approximates an RFunc as a quadratic
// function centered at x0 and evaluates the new
// approximation at x.
// The result becomes invalid if x0 is modified.
// The given function must output a scaler, like
// pretty much any loss function does.
func QuadApprox(f autofunc.RFunc, x0 linalg.Vector, x autofunc.Result) autofunc.Result {
	return &quadApproxResult{
		lazyQuadApprox: lazyQuadApprox{
			X0: x0,
			X:  x.Output(),
			F:  f,
		},
		Input: x,
	}
}

// QuadApproxR is like QuadApprox, but with R-operators.
func QuadApproxR(f autofunc.RFunc, x0 linalg.Vector, x autofunc.RResult) autofunc.RResult {
	return &quadApproxRResult{
		lazyQuadApprox: lazyQuadApprox{
			X0: x0,
			X:  x.Output(),
			XR: x.ROutput(),
			F:  f,
		},
		Input: x,
	}
}

type lazyQuadApprox struct {
	Lock sync.Mutex

	X0 linalg.Vector
	X  linalg.Vector
	XR linalg.Vector
	F  autofunc.RFunc

	Displacement linalg.Vector

	EvalResult   autofunc.RResult
	EvalHessProd linalg.Vector
	EvalGrad     linalg.Vector
	OutputVec    linalg.Vector

	REvalResult   autofunc.RResult
	REvalHessProd linalg.Vector
	ROutputVec    linalg.Vector
}

func (q *lazyQuadApprox) Output() linalg.Vector {
	q.Lock.Lock()
	defer q.Lock.Unlock()
	if q.EvalResult == nil {
		q.evaluate()
	}
	if q.OutputVec == nil {
		hessDot := 0.5 * q.EvalHessProd.Dot(q.Displacement)
		q.OutputVec = []float64{q.EvalResult.Output()[0] +
			q.EvalResult.ROutput()[0] + hessDot}
	}
	return q.OutputVec
}

func (q *lazyQuadApprox) ROutput() linalg.Vector {
	q.Lock.Lock()
	defer q.Lock.Unlock()
	if q.REvalResult == nil {
		q.evaluateR()
	}
	if q.ROutputVec == nil {
		if q.Displacement == nil {
			q.computeDisplacement()
		}
		hessDot := q.Displacement.Dot(q.REvalHessProd)
		q.ROutputVec = []float64{q.REvalResult.ROutput()[0] + hessDot}
	}
	return q.ROutputVec
}

func (q *lazyQuadApprox) evaluate() {
	if q.Displacement == nil {
		q.computeDisplacement()
	}
	inVar := &autofunc.Variable{Vector: q.X0}
	inRVar := &autofunc.RVariable{
		Variable:   inVar,
		ROutputVec: q.Displacement,
	}
	q.EvalResult = q.F.ApplyR(autofunc.RVector{}, inRVar)

	if len(q.EvalResult.Output()) != 1 {
		panic("QuadApprox only works with single-output functions")
	}

	variables := []*autofunc.Variable{inVar}
	grad := autofunc.NewGradient(variables)
	rgrad := autofunc.NewRGradient(variables)
	q.EvalResult.PropagateRGradient([]float64{1}, []float64{0}, rgrad, grad)

	q.EvalHessProd = rgrad[inVar]
	q.EvalGrad = grad[inVar]
}

func (q *lazyQuadApprox) evaluateR() {
	inVar := &autofunc.Variable{Vector: q.X0}
	inRVar := &autofunc.RVariable{
		Variable:   inVar,
		ROutputVec: q.XR,
	}
	q.REvalResult = q.F.ApplyR(autofunc.RVector{}, inRVar)

	if len(q.REvalResult.Output()) != 1 {
		panic("QuadApprox only works with single-output functions")
	}

	variables := []*autofunc.Variable{inVar}
	rgrad := autofunc.NewRGradient(variables)
	q.REvalResult.PropagateRGradient([]float64{1}, []float64{0}, rgrad, nil)
	q.REvalHessProd = rgrad[inVar]
}

func (q *lazyQuadApprox) computeDisplacement() {
	q.Displacement = q.X.Copy().Add(q.X0.Copy().Scale(-1))
}

type quadApproxResult struct {
	lazyQuadApprox

	Input autofunc.Result
}

func (q *quadApproxResult) Constant(g autofunc.Gradient) bool {
	return q.Input.Constant(g)
}

func (q *quadApproxResult) PropagateGradient(upstream linalg.Vector, g autofunc.Gradient) {
	if q.Input.Constant(g) {
		return
	}
	q.Lock.Lock()
	if q.EvalResult == nil {
		q.evaluate()
	}
	downstream := q.EvalGrad.Copy().Add(q.EvalHessProd).Scale(upstream[0])
	q.Lock.Unlock()
	q.Input.PropagateGradient(downstream, g)
}

type quadApproxRResult struct {
	lazyQuadApprox

	Input autofunc.RResult
}

func (q *quadApproxRResult) Constant(rg autofunc.RGradient, g autofunc.Gradient) bool {
	return q.Input.Constant(rg, g)
}

func (q *quadApproxRResult) PropagateRGradient(upstream, upstreamR linalg.Vector,
	rg autofunc.RGradient, g autofunc.Gradient) {
	if q.Input.Constant(rg, g) {
		return
	}
	q.Lock.Lock()
	if q.EvalResult == nil {
		q.evaluate()
	}
	if q.REvalResult == nil {
		q.evaluateR()
	}
	rawGradient := q.EvalGrad.Copy().Add(q.EvalHessProd)
	downstream := rawGradient.Copy().Scale(upstream[0])
	gradDeriv := q.REvalHessProd.Copy()
	downstreamR := rawGradient.Scale(upstreamR[0]).Add(gradDeriv.Scale(upstream[0]))
	q.Lock.Unlock()
	q.Input.PropagateRGradient(downstream, downstreamR, rg, g)
}
