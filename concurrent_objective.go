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
	// quadratic approximation and adds it to gradOut.
	//
	// Since this is additive, you may want to ensure that
	// vecOut is initially set all 0's.
	QuadGrad(delta ConstParamDelta, s sgd.SampleSet, gradOut ConstParamDelta)

	// QuadHessian applies the objective's approximation's
	// Hessian to the delta and adds the result to vecOut.
	//
	// Since this is additive, you may want to ensure that
	// vecOut is initially set all 0's.
	QuadHessian(delta ConstParamDelta, s sgd.SampleSet, vecOut ConstParamDelta)
}

// Objective is an objective function which can be truly
// evaluated or evaluated via a quadratic approximation.
type Objective interface {
	QuadObjective

	// Objective evaluates the true objective (rather than
	// its approximation) at the given delta.
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

	deltaCache deltaCache
}

func (c *ConcurrentObjective) Quad(delta ConstParamDelta, s sgd.SampleSet) float64 {
	return c.sumValues(func(subSet sgd.SampleSet) float64 {
		return c.Wrapped.Quad(delta, subSet)
	}, s)
}

func (c *ConcurrentObjective) QuadGrad(delta ConstParamDelta,
	s sgd.SampleSet, sumOut ConstParamDelta) {
	c.sumDeltas(func(subSet sgd.SampleSet, out ConstParamDelta) {
		c.Wrapped.QuadGrad(delta, subSet, out)
	}, s, sumOut)
}

func (c *ConcurrentObjective) QuadHessian(delta ConstParamDelta,
	s sgd.SampleSet, sumOut ConstParamDelta) {
	c.sumDeltas(func(subSet sgd.SampleSet, out ConstParamDelta) {
		c.Wrapped.QuadHessian(delta, subSet, out)
	}, s, sumOut)
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

func (c *ConcurrentObjective) sumDeltas(r func(sgd.SampleSet, ConstParamDelta),
	s sgd.SampleSet, sumOut ConstParamDelta) {
	sampleChan := c.subBatchChan(s)
	deltaChan := make(chan ConstParamDelta, c.goroutineCount())

	wg := c.runGoroutines(func() {
		sum := c.deltaCache.Alloc(sumOut.variables())
		for subSet := range sampleChan {
			r(subSet, sum)
		}
		deltaChan <- sum
	})
	go func() {
		wg.Wait()
		close(deltaChan)
	}()

	for delta := range deltaChan {
		sumOut.addDelta(delta, 1)
		c.deltaCache.Release(delta)
	}
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
