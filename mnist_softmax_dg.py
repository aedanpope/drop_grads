# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# ORIGINAL: https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_softmax.py
# Modifications by Aedan Pope

"""A very simple MNIST classifier.
See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np
import random
import argparse

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

out_file

def outp(s):
  print(s)
  if out_file:
    print >>out_file, s
    file.flush(out_file)


def main(_):


  parser = argparse.ArgumentParser('Run MNIST drop_grads experiment.')
  parser.add_argument('-f', '--out_file', default='results.txt'
      help='File to append results to.')
  parser.add_argument('-p', '--drop_grads_prob', type=float, default=0.5,
      help='Probability of dropping grads for each variable in each training batch.')
  parser.add_argument('-b', '--batches', type=int, default=2000,
      help='Number of batches of size 50 to train.')
  args = parser.parse_args()

  drop_grads_prob = args.drop_grads_prob
  print("drop_grads_prob = %d"%drop_grads_prob)


  out_file = open(args.out_file, 'a')
  print >>out_file, ""
  print >>out_file, "Command:"
  print >>out_file, str(' '.join(sys.argv))

  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  x_image = tf.reshape(x, [-1,28,28,1])

  W_conv1 = weight_variable([5, 5, 1, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  W_fc1 = weight_variable([7 * 7 * 64, 1024])
  b_fc1 = bias_variable([1024])
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  W_fc2 = weight_variable([1024, 10])
  b_fc2 = bias_variable([10])
  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy_loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

  # Variables to calculate the gradients seprately before applying them.

  tvars = tf.trainable_variables()
  var_grad_holders = []
  for idx,var in enumerate(tvars):
    placeholder = tf.placeholder(tf.float32,name=str(idx)+'_holder')
    var_grad_holders.append(placeholder)
  # Run this to get the gradients that should be applied given some training data.
  run_calc_grads = tf.gradients(cross_entropy_loss, tvars)

  adam = tf.train.AdamOptimizer(1e-4)
  # To apply gradients, run this with a feed_dict that populates var_grad_holders
  # (probably getting gradients from run_calc_grads)
  run_update_grads = adam.apply_gradients(zip(var_grad_holders, tvars))

  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  run_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  for i in range(args.batches+1):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
      train_accuracy = run_accuracy.eval(feed_dict={
          x:batch[0], y_: batch[1], keep_prob: 1.0})
      test_accuracy = run_accuracy.eval(feed_dict={
          x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
      outp("step %d, training accuracy %g, test accuracy %g"%(i, train_accuracy, test_accuracy))

    grads = sess.run(run_calc_grads, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    grad_dict = dict(zip(var_grad_holders, grads))

    # Zero out the gradients computed from this training batch
    # for some variables randomly chosen.
    for var in var_grad_holders:
      if (random.random() < drop_grads_prob):
        grad_dict[var] = np.zeros(grad_dict[var].shape)
    run_update_grads.run(feed_dict=grad_dict)

  outp("test accuracy %g"%run_accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)