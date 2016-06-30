package hessfree

import (
	"runtime"
	"sync"

	"github.com/unixpickle/sgd"
)

const defaultMaxSubBatch = 15

// Objective is a quadratic objective function that
// takes ConstParamDeltas as arguments.
type Objective interface {
	// ObjectiveHessian applies the objective's Hessian
	// to the delta.
	ObjectiveHessian(delta ConstParamDelta, s sgd.SampleSet) ConstParamDelta

	// ObjectiveGrad computes the objective's gradient at
	// the delta.
	ObjectiveGrad(delta ConstParamDelta, s sgd.SampleSet) ConstParamDelta

	// Objective evaluates the objective at the delta.
	Objective(delta ConstParamDelta, s sgd.SampleSet) float64
}

// ConcurrentObjective is an Objective which wraps
// another Objective and parallelizes calls to that
// Objective while ensuring that no extremely large
// batches are passed to the Objective at once.
type ConcurrentObjective struct {
	// Wrapped is the wrapped Objective.
	Wrapped Objective

	// MaxConcurrency is the maximum number of goroutines
	// on which the wrapped Objective can be used at once.
	// If this is 0, GOMAXPROCS is used.
	MaxConcurrency int

	// MaxSubBatch is the maximum number of samples that
	// can be passed to the wrapped Objective at once.
	// If this is 0, a reasonable default is used.
	MaxSubBatch int
}

func (c *ConcurrentObjective) Objective(delta ConstParamDelta, s sgd.SampleSet) float64 {
	sampleChan := c.subBatchChan(s)

	var res float64
	var resLock sync.Mutex

	c.runGoroutines(func() {
		for subSet := range sampleChan {
			output := c.Wrapped.Objective(delta, subSet)
			resLock.Lock()
			res += output
			resLock.Unlock()
		}
	}).Wait()

	return res
}

func (c *ConcurrentObjective) ObjectiveGrad(delta ConstParamDelta,
	s sgd.SampleSet) ConstParamDelta {
	return c.sumDeltas(func(subSet sgd.SampleSet) ConstParamDelta {
		return c.Wrapped.ObjectiveGrad(delta, subSet)
	}, s)
}

func (c *ConcurrentObjective) ObjectiveHessian(delta ConstParamDelta,
	s sgd.SampleSet) ConstParamDelta {
	return c.sumDeltas(func(subSet sgd.SampleSet) ConstParamDelta {
		return c.Wrapped.ObjectiveHessian(delta, subSet)
	}, s)
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
		}
		for variable, v := range delta {
			resVec := res[variable]
			resVec.Add(v)
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

	return res
}

func (c *ConcurrentObjective) goroutineCount() int {
	if c.MaxConcurrency != 0 {
		return c.MaxConcurrency
	} else {
		return runtime.GOMAXPROCS(0)
	}
}
