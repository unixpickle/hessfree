package hessfree

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A ParamDelta is a displacement vector (t - t0) where
// t contains new variable values and t0 contains the
// current value of those variables.
type ParamDelta map[*autofunc.Variable]autofunc.Result

// outputRVector returns an RVector mapping variables in
// p to the corresponding RResults' outputs.
func (p ParamDelta) outputRVector() autofunc.RVector {
	res := autofunc.RVector{}
	for variable, r := range p {
		res[variable] = r.Output()
	}
	return res
}

// zeroGradient produces an autofunc.Gradient with zero
// vectors for all of the delta's variables.
func (p ParamDelta) zeroGradient() autofunc.Gradient {
	res := autofunc.Gradient{}
	for variable := range p {
		res[variable] = make(linalg.Vector, len(variable.Vector))
	}
	return res
}

// A ParamRDelta is like a ParamDelta, but each delta
// has a derivative with respect to a variable R.
type ParamRDelta map[*autofunc.Variable]autofunc.RResult

// outputRVector returns an RVector mapping variables in
// p to the corresponding RResults' outputs.
func (p ParamRDelta) outputRVector() autofunc.RVector {
	res := autofunc.RVector{}
	for variable, r := range p {
		res[variable] = r.Output()
	}
	return res
}

// rOutputRVector is like outputRVector, but it uses the
// RResults' ROutput instead of their Output.
func (p ParamRDelta) rOutputRVector() autofunc.RVector {
	res := autofunc.RVector{}
	for variable, r := range p {
		res[variable] = r.ROutput()
	}
	return res
}

// zeroGradient produces an autofunc.Gradient with zero
// vectors for all of the delta's variables.
func (p ParamRDelta) zeroGradient() autofunc.Gradient {
	res := autofunc.Gradient{}
	for variable := range p {
		res[variable] = make(linalg.Vector, len(variable.Vector))
	}
	return res
}

// A ConstParamDelta is like a ParamDelta, except that
// the delta vectors are constant vectors.
type ConstParamDelta map[*autofunc.Variable]linalg.Vector

// addToVars adds the delta to its underlying variables.
func (c ConstParamDelta) addToVars() {
	for variable, delta := range c {
		variable.Vector.Add(delta)
	}
}

// magSquared returns the squared magnitude of the delta.
func (c ConstParamDelta) magSquared() float64 {
	return c.dot(c)
}

// dot returns the dot product of two deltas.
func (c ConstParamDelta) dot(c1 ConstParamDelta) float64 {
	var res float64
	for v, x := range c {
		res += x.DotFast(c1[v])
	}
	return res
}

// copy returns a copy of this delta.
func (c ConstParamDelta) copy() ConstParamDelta {
	res := ConstParamDelta{}
	for v, x := range c {
		res[v] = make(linalg.Vector, len(x))
		copy(res[v], x)
	}
	return res
}

// scale scales the delta by the given scaler.
func (c ConstParamDelta) scale(scaler float64) {
	for _, x := range c {
		x.Scale(scaler)
	}
}

// addDelta adds a scaled version of the given delta
// to this delta.
func (c ConstParamDelta) addDelta(c1 ConstParamDelta, scaler float64) {
	// TODO: optimize this using BLAS.
	for v, x := range c {
		x.Add(c1[v].Copy().Scale(scaler))
	}
}
