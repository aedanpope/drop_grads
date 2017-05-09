# Effective learning while misplacing gradient updates.

## Overview

Consider executing BackProp on a large computation graph in a distributed architecture, where a single back-propagation through the computational graph takes place over multiple machines. Suppose access to individual machines is unreliable, and may be revoked at any time. How can we make meaningful training progress in such an environment?

We experiment with dropping the gradients for random variables from the computational graph in the execution of each instance of backpropagation, and show that effective learning is still possible by naively just training with the subset of gradients that are still available.

## Rationale

Given some gradients computed by backpropagation, applying those gradients to the variables in a network will reduce the Loss for the input training data used to generate those gradients. Applying a single gradient to a single variable in the computational network will also reduce the loss. Therefore, it seems worthwhile to apply any gradients available after a partially complete backprop computation, even if other gradients were not able to be computed.

## Method

Setup is the standard MNIST problem with typical high-performance CNN from the [TensorFlow tutorial](https://www.tensorflow.org/get_started/mnist/pros). Typically this achieves an accuracy of around 99.2% accuracy on MNIST.

Modified backpro Algorithm:
1. Feed a batch of 50 images to the network, and run backprop to compute gradients.
2. For each variable in the network (e.g. one variable being the weights or biases for some layer in the network), with `(1-keep_prob)`, we zero the gradient for that variable.
3. Apply the gradients (some of which are zeor) to train the network.

## Results

After 20k batches, keeping 50%+ of the variable-gradients achieves the near optimal performance of 99.2% accuracy, whilst only keeping 20% of gradients has only achieved an accuracy of about 89.9%.
![Graph of results](results_raw.png)

Consider that if only 50% of the computed variable-gradients are applied, then in effect for a given number of batches the network has received half as much training. Intuitively, we'd expect `keep_prob=0.5` to take around twice as long to train as `keep_prob=1.0`. "pro-rating" the number of batches by multiplying by keep_prob encodes this, and in the below graph we can see that `keep_prob=0.5` is able to train a little bit faster than twice-as-long as `keep_prob=1.0`.
![Graph of results prorated](results_prorated.png)
