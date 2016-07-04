package hessfree

import (
	"math"

	"github.com/unixpickle/sgd"
)

const (
	defaultConvergenceMinK    = 10
	defaultConvergenceKScale  = 0.1
	defaultConvergenceEpsilon = 0.0005
	defaultBacktrackRate      = 1.3
)

// ConvergenceCriteria stores the parameters for the
// relative change convergence criteria described in
// Martens (2010).
// If the values are 0, defaults from Martens (2010)
// are used.
type ConvergenceCriteria struct {
	MinK    float64
	KScale  float64
	Epsilon float64
}

// A Trainer runs Hessian Free on a Learner.
type Trainer struct {
	// Learner is trained using Hessian Free.
	Learner Learner

	// Samples contains all of the training samples.
	Samples sgd.SampleSet

	// BatchSize is the size of mini-batches.
	BatchSize int

	// UI is the means by which the Trainer communicates with
	// the user, logging information and receiving termination
	// signals.
	UI UI

	// Convergence are the convergence criteria.
	Convergence ConvergenceCriteria

	// BacktrackRate is a constant greater than 1 which controls
	// how frequently backtracking checkpoints are made.
	// If this is 0, the default from Martens (2010) is used.
	BacktrackRate float64
}

func (t *Trainer) Train() {
	var epoch int
	var lastSolution ConstParamDelta
	var cache deltaCache
	for {
		shuffled := t.Samples.Copy()
		sgd.ShuffleSampleSet(shuffled)

		var miniBatch int
		for i := 0; i < shuffled.Len(); i += t.BatchSize {
			bs := t.BatchSize
			if bs > shuffled.Len()-i {
				bs = shuffled.Len() - i
			}
			subset := shuffled.Subset(i, i+bs)
			if t.UI.ShouldStop() {
				return
			}
			t.UI.LogNewMiniBatch(epoch, miniBatch)

			solver := cgSolver{
				Trainer:   t,
				Objective: t.Learner.MakeObjective(),
				Samples:   subset,
				Solution:  lastSolution,
				Cache:     cache,
			}
			for solver.Step() {
				if t.UI.ShouldStop() {
					return
				}
			}
			useDelta := solver.Best()
			lastSolution = solver.Solution
			t.Learner.Adjust(useDelta, subset)
			solver.Release()

			miniBatch++
		}
		epoch++
	}
}

type cgSolver struct {
	Trainer   *Trainer
	Objective Objective
	Samples   sgd.SampleSet
	Solution  ConstParamDelta
	Cache     deltaCache

	residual          ConstParamDelta
	projectedResidual ConstParamDelta
	residualMag2      float64

	justBacktracked bool
	backtrackCount  int
	backtrackDeltas []ConstParamDelta
	backtrackValues []float64

	startObjective float64
	quadValues     []float64
}

// Step runs a step of CG and returns true if another
// step is desired (i.e. no termination).
func (c *cgSolver) Step() (shouldContinue bool) {
	c.initializeIfNeeded()

	projHessian := c.allocDelta()
	defer c.Cache.Release(projHessian)
	c.Objective.QuadHessian(c.projectedResidual, c.Samples, projHessian)

	projHessianMag := c.projectedResidual.dot(projHessian)
	if projHessianMag == 0 || c.residualMag2 == 0 {
		return false
	}

	c.justBacktracked = false
	stepSize := c.residualMag2 / projHessianMag

	c.Solution.addDelta(c.projectedResidual, stepSize)

	quadOutput := c.Objective.Quad(c.Solution, c.Samples)
	c.quadValues = append(c.quadValues, quadOutput)

	c.Trainer.UI.LogCGIteration(stepSize, quadOutput)

	if c.converging() {
		return false
	}

	oldRMag2 := c.residualMag2
	c.residual.addDelta(projHessian, -stepSize)
	c.residualMag2 = c.residual.magSquared()

	beta := c.residualMag2 / oldRMag2
	c.projectedResidual.scale(beta)
	c.projectedResidual.addDelta(c.residual, 1)

	c.updateBacktracking()

	return true
}

// Best returns the best known solution, including the
// current solution and all the backtracked ones.
func (c *cgSolver) Best() ConstParamDelta {
	var bestVal float64
	var bestDelta ConstParamDelta
	for i, v := range c.backtrackValues {
		if v < bestVal || i == 0 {
			bestDelta = c.backtrackDeltas[i]
			bestVal = v
		}
	}
	if !c.justBacktracked {
		btValue := c.Objective.Objective(c.Solution, c.Samples)
		if btValue < bestVal || bestDelta == nil {
			return c.Solution
		}
	}
	return bestDelta
}

// Release releases all the deltas back to the cache
// except for the previous solution.
func (c *cgSolver) Release() {
	for _, tempDelta := range c.backtrackDeltas {
		c.Cache.Release(tempDelta)
	}
	c.Cache.Release(c.residual)
	c.Cache.Release(c.projectedResidual)
}

func (c *cgSolver) initializeIfNeeded() {
	if c.Solution == nil {
		// Will only happen in the first CG run, since
		// information sharing is used.
		c.Solution = c.allocDelta()
	}

	if c.residual == nil {
		c.residual = c.allocDelta()
		c.Objective.QuadGrad(c.Solution, c.Samples, c.residual)
		c.residual.scale(-1)
		c.projectedResidual = c.allocDelta()
		c.projectedResidual.copy(c.residual)

		c.residualMag2 = c.residual.magSquared()
		c.startObjective = c.Objective.Objective(ConstParamDelta{}, c.Samples)

		c.Trainer.UI.LogCGStart(c.Objective.Quad(c.Solution, c.Samples), c.startObjective)
	}
}

func (c *cgSolver) converging() bool {
	if len(c.quadValues) < 2 || c.quadValues[len(c.quadValues)-1] > c.startObjective {
		return false
	}

	kScale := c.Trainer.Convergence.KScale
	minK := c.Trainer.Convergence.MinK
	eps := c.Trainer.Convergence.Epsilon
	if kScale == 0 {
		kScale = defaultConvergenceKScale
	}
	if minK == 0 {
		minK = defaultConvergenceMinK
	}
	if eps == 0 {
		eps = defaultConvergenceEpsilon
	}

	k := int(math.Max(minK, kScale*float64(len(c.quadValues))))
	if k >= len(c.quadValues) {
		return false
	}

	currentImprovement := (c.quadValues[len(c.quadValues)-1] - c.startObjective)
	oldImprovement := (c.quadValues[len(c.quadValues)-1-k] - c.startObjective)
	return (currentImprovement-oldImprovement)/currentImprovement < float64(k)*eps
}

func (c *cgSolver) updateBacktracking() {
	doneIters := len(c.quadValues)
	btRate := c.Trainer.BacktrackRate
	if btRate == 0 {
		btRate = defaultBacktrackRate
	}
	expValue := math.Pow(btRate, float64(c.backtrackCount))
	if int(expValue) > doneIters {
		return
	}
	for int(math.Pow(btRate, float64(c.backtrackCount))) <= doneIters {
		c.backtrackCount++
	}

	btValue := c.Objective.Objective(c.Solution, c.Samples)
	savedSolution := c.allocDelta()
	savedSolution.copy(c.Solution)
	c.backtrackDeltas = append(c.backtrackDeltas, savedSolution)
	c.backtrackValues = append(c.backtrackValues, btValue)
	c.justBacktracked = true
}

func (c *cgSolver) allocDelta() ConstParamDelta {
	return c.Cache.Alloc(c.Trainer.Learner.Parameters())
}
