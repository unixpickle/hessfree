package hessfree

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// QuadApprox approximates a Func as a quadratic
// function centered at x0 and evaluates the new
// approximation at x.
// The given function must output a scalers, like
// pretty much any loss function.
func QuadApprox(b autofunc.Func, x0 linalg.Vector, x autofunc.Result) autofunc.Result {
	// TODO: this.
	panic("not yet implemented")
}

// QuadApproxR is like QuadApprox, but with R-operators.
func QuadApproxR(b autofunc.RFunc, x0 linalg.Vector, x autofunc.RResult) autofunc.RResult {
	// TODO: this.
	panic("not yet implemented")
}
