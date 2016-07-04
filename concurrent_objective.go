package hessfree

import (
	"runtime"
	"sync"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
)

const defaultMaxSubBatch = 15

// QuadObjective represents a quadratic approximation
// of some objective function.
type QuadObjective interface {
	// Quad evaluates the objective's approximation.
	Quad(delta ConstParamDelta, s sgd.SampleSet) float64

	// QuadGrad computes the gradient of the objective's
	// quadratic approximation.
	QuadGrad(delta ConstParamDelta, s sgd.SampleSet) ConstParamDelta

	// QuadHessian applies the objective's approximation's
	// Hessian to the delta while simultaneously evaluating
	// the approximation at x.
	// This returns the result of the Hessian application
	// and the value of the approximation at x.
	QuadHessian(delta, x ConstParamDelta, s sgd.SampleSet) (ConstParamDelta, float64)
}

// Objective is an objective function which can be truly
// evaluated or evaluated via a quadratic approximation.
type Objective interface {
	QuadObjective

	// Objective evaluates the true objective (rather than
	// its approximation) at the given delta.
	// The delta may be empty (i.e. contain no keys), in
	// which case the objective is evaluated at offset=0.
	Objective(delta ConstParamDelta, s sgd.SampleSet) float64
}

// WrappedObjective is like an Objective, but the true
// objective can only be evaluated with a delta of 0.
type WrappedObjective interface {
	QuadObjective

	// ObjectiveAtZero evaluates the true objective using
	// the current values of the underlying variables.
	ObjectiveAtZero(s sgd.SampleSet) float64
}

// ConcurrentObjective is an Objective which wraps
// a WrappedObjective and parallelizes calls to that
// objective while ensuring that no extremely large
// batches are passed to the objective at once.
type ConcurrentObjective struct {
	// Wrapped is the wrapped objective.
	//
	// If MaxConcurrency is not set or is set to a
	// value greater than 1, then the objective's
	// methods must be concurrency-safe.
	Wrapped WrappedObjective

	// MaxConcurrency is the maximum number of goroutines
	// on which the wrapped Objective can be used at once.
	// If this is 0, GOMAXPROCS is used.
	MaxConcurrency int

	// MaxSubBatch is the maximum number of samples that
	// can be passed to the wrapped Objective at once.
	// If this is 0, a reasonable default is used.
	MaxSubBatch int
}

func (c *ConcurrentObjective) Quad(delta ConstParamDelta, s sgd.SampleSet) float64 {
	return c.sumValues(func(subSet sgd.SampleSet) float64 {
		return c.Wrapped.Quad(delta, subSet)
	}, s)
}

func (c *ConcurrentObjective) QuadGrad(delta ConstParamDelta,
	s sgd.SampleSet) ConstParamDelta {
	return c.sumDeltas(func(subSet sgd.SampleSet) ConstParamDelta {
		return c.Wrapped.QuadGrad(delta, subSet)
	}, s)
}

func (c *ConcurrentObjective) QuadHessian(delta, x ConstParamDelta,
	s sgd.SampleSet) (ConstParamDelta, float64) {
	var yLock sync.Mutex
	var y float64
	deltaSum := c.sumDeltas(func(subSet sgd.SampleSet) ConstParamDelta {
		res, localY := c.Wrapped.QuadHessian(delta, x, subSet)
		yLock.Lock()
		y += localY
		yLock.Unlock()
		return res
	}, s)
	return deltaSum, y
}

func (c *ConcurrentObjective) Objective(delta ConstParamDelta, s sgd.SampleSet) float64 {
	backups := map[*autofunc.Variable]linalg.Vector{}
	if delta != nil {
		for variable, newVec := range delta {
			backups[variable] = variable.Vector
			variable.Vector = variable.Vector.Copy().Add(newVec)
		}
	}
	res := c.sumValues(func(subSet sgd.SampleSet) float64 {
		return c.Wrapped.ObjectiveAtZero(subSet)
	}, s)
	for variable, backup := range backups {
		variable.Vector = backup
	}
	return res
}

func (c *ConcurrentObjective) sumValues(r func(sgd.SampleSet) float64, s sgd.SampleSet) float64 {
	sampleChan := c.subBatchChan(s)

	var res float64
	var resLock sync.Mutex

	c.runGoroutines(func() {
		for subSet := range sampleChan {
			output := r(subSet)
			resLock.Lock()
			res += output
			resLock.Unlock()
		}
	}).Wait()

	return res
}

func (c *ConcurrentObjective) sumDeltas(r func(sgd.SampleSet) ConstParamDelta,
	s sgd.SampleSet) ConstParamDelta {
	sampleChan := c.subBatchChan(s)

	var res ConstParamDelta
	deltaChan := make(chan ConstParamDelta, c.goroutineCount())

	wg := c.runGoroutines(func() {
		for subSet := range sampleChan {
			deltaChan <- r(subSet)
		}
	})
	go func() {
		wg.Wait()
		close(deltaChan)
	}()

	for delta := range deltaChan {
		if res == nil {
			res = delta
		} else {
			for variable, v := range delta {
				resVec := res[variable]
				resVec.Add(v)
			}
		}
	}

	return res
}

func (c *ConcurrentObjective) runGoroutines(r func()) *sync.WaitGroup {
	wg := &sync.WaitGroup{}

	for i := 0; i < c.goroutineCount(); i++ {
		wg.Add(1)
		go func() {
			r()
			wg.Done()
		}()
	}

	return wg
}

func (c *ConcurrentObjective) subBatchChan(s sgd.SampleSet) <-chan sgd.SampleSet {
	subSize := c.MaxSubBatch
	if subSize == 0 {
		subSize = defaultMaxSubBatch
	}

	batchCount := s.Len()/subSize + 1
	res := make(chan sgd.SampleSet, batchCount)

	for i := 0; i < s.Len(); i += subSize {
		bs := subSize
		if bs > s.Len()-i {
			bs = s.Len() - i
		}
		res <- s.Subset(i, i+bs)
	}
	close(res)

	return res
}

func (c *ConcurrentObjective) goroutineCount() int {
	if c.MaxConcurrency != 0 {
		return c.MaxConcurrency
	} else {
		return runtime.GOMAXPROCS(0)
	}
}
