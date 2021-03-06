# hessfree

Package hessfree implements the [Hessian Free](http://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf) 2nd order optimization algorithm. This algorithm has proven very successful for training deep neural networks and RNNs.

I aim to make this package as general as possible. However, Hessian Free is much more involved than Stochastic Gradient Descent, sometimes requiring structural information about the underlying function it is optimizing (e.g. [Sutskever's structural damping](http://www.cs.utoronto.ca/~ilya/pubs/2011/HF-RNN.pdf)).
