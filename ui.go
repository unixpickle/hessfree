package hessfree

import (
	"fmt"
	"log"
	"os"
	"os/signal"
	"sync/atomic"
)

// A UI logs information about a Hessian Free training
// session and accepts input from the user.
type UI interface {
	LogCGIteration(stepSize, quadValue float64)
	LogNewMiniBatch(epochNumber, batchNumber int)
	ShouldStop() bool
}

// ConsoleUI is a UI which outputs things to the console
// using the log package and stops when the user sends a
// kill interrupt.
type ConsoleUI struct {
	killFlag uint32
}

func NewConsoleUI() *ConsoleUI {
	res := &ConsoleUI{}

	go func() {
		c := make(chan os.Signal, 1)
		signal.Notify(c, os.Interrupt)
		<-c
		signal.Stop(c)
		close(c)
		atomic.StoreUint32(&res.killFlag, 1)
		fmt.Println("\nCaught interrupt. Ctrl+C again to terminate.")
	}()

	return res
}

func (c *ConsoleUI) LogCGIteration(stepSize, quadValue float64) {
	log.Printf("CG iteration (stepSize=%f, quad=%f)", stepSize, quadValue)
}

func (c *ConsoleUI) LogNewMiniBatch(epochNum, batchNum int) {
	log.Printf("Next mini-batch (epoch=%d, batch=%d)", epochNum, batchNum)
}

func (c *ConsoleUI) ShouldStop() bool {
	return atomic.LoadUint32(&c.killFlag) != 0
}
