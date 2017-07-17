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

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import os
from checkpoint import list_variables
FLAGS = None
import matplotlib.pyplot as plt





def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  name_1="conv1"
  W_conv1 = weight_variable([5, 5, 1, 32],name_1+"_filters")
  b_conv1 = bias_variable([32],name_1+"_biases")
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  name_2="conv2"
  W_conv2 = weight_variable([5, 5, 32, 64],name_2+"_filters")
  b_conv2 = bias_variable([64],name_2+"_biases")
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  name_3="fc1"
  W_fc1 = weight_variable([7 * 7 * 64, 1024],name_3+"_weights")
  b_fc1 = bias_variable([1024],name_3+"_biases")

  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # add a new fc layer
  name_add ="fc_add"
  W_fc_add = weight_variable([1024, 1024],name_add+"_weights")
  b_fc_add = bias_variable([1024],name_add+"_biases")

  h_fc_add = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc_add) + b_fc_add)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  h_fc_add_drop = tf.nn.dropout(h_fc_add, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit new output layer
  name_4="fc_out"
  W_fc2 = weight_variable([1024, 10],name_4+"_weights")
  b_fc2 = bias_variable([10],name_4+"_biases")

  # Map the 1024 features to 10 classes, one for each digit
  # name_4="fc2"
  # W_fc2 = weight_variable([1024, 10],name_4+"_weights")
  # b_fc2 = bias_variable([10],name_4+"_biases")
  #
  # y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  y_conv = tf.matmul(h_fc_add_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape,name):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.get_variable(name=name, initializer=initial)


def bias_variable(shape,name):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.get_variable(name=name, initializer=initial)


def main(_):
  os.environ["CUDA_VISIBLE_DEVICES"] = "1"
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)



  # Build the graph for the deep net
  with tf.variable_scope("cnn") as scope:
    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])

    y_conv, keep_prob = deepnn(x)

    with tf.Session() as sess:

        # init all variables in the new model
        # sess.run(tf.global_variables_initializer())
        all_val = tf.get_collection(tf.GraphKeys.VARIABLES)
        print(len(all_val))
        for v in all_val:
            print(v.name)


        # get the list of the names of the variables need to restored
        checkpoint = "records/all/model.ckpt"
        restored_val = list_variables(checkpoint)
        print(len(restored_val))
        print(restored_val)

        # build the dictionary to restore the variables of the new model that are needed from checkpoints
        val_dic = {}
        val_list_old=[]
        val_list_new=[]
        for v in all_val:
            if v.name.split(':')[0] in restored_val:
                val_dic[v.name.split(':')[0]] = v
                val_list_old.append(v)
            else:
                val_list_new.append(v)
        print("old val")
        for v in val_list_old:
            print(v.name)
        print("new val")
        for v in val_list_new:
            print(v.name)

        # var_test = [v for v in all_val if v.name == "cnn/conv1_biases:0"][0]
        # print (var_test.name)
        # print (var_test.eval())

        # cross_entropy = tf.reduce_mean(
        #     tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        # train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
        # correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # tf.summary.scalar("accuracy",accuracy)

        cross_entropy=tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        opt1=tf.train.AdamOptimizer(1e-8)
        opt2=tf.train.AdadeltaOptimizer(1e-5)
        grads=tf.gradients(cross_entropy,val_list_old+val_list_new)
        grads1=grads[:len(val_list_old)]
        grads2=grads[len(val_list_old):]
        train_op1=opt1.apply_gradients(zip(grads1,val_list_old))
        train_op2=opt2.apply_gradients(zip(grads2,val_list_new))
        train_op=tf.group(train_op1,train_op2)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy",accuracy)

        sess.run(tf.initialize_all_variables())

        all_val_1 = tf.get_collection(tf.GraphKeys.VARIABLES)
        for v in all_val_1:
            print(v.name)

        saver = tf.train.Saver(val_dic)

        saver.restore(sess, checkpoint)

        all_val_2 = tf.get_collection(tf.GraphKeys.VARIABLES)
        print(len(all_val_2))
        for v in all_val_2:
            print(v.name)
            print(v.eval())

        for i in range(2000):
          batch_xs, batch_ys = mnist.train.next_batch(50)

          if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            train_op.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

      # print('test accuracy %g' % accuracy.eval(feed_dict={
      #   x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

        save_path = saver.save(sess,"records/addlayer/model.ckpt")
        print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
