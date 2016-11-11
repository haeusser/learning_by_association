"""
Copyright 2016 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Definitions and utilities for the MNIST model.

This file contains functions that are needed for semisup training and evalutaion
on the MNIST dataset.
They are used in MNIST_train_eval.py.

"""

import gzip
import numpy as np

import tensorflow as tf
import tf.contrib.slim as slim
from tf.python.platform import gfile
import semisup


NUM_LABELS = 10
IMAGE_SHAPE = [28, 28, 1]


def get_data(name):
  """Utility for convenient data loading."""
  datadir = 'path_to_mnist'

  if name == 'train':
    return extract_images(datadir +
                          '/train-images-idx3-ubyte.gz'), extract_labels(
                              datadir + '/train-labels-idx1-ubyte.gz')
  elif name == 'test':
    return extract_images(datadir +
                          '/t10k-images-idx3-ubyte.gz'), extract_labels(
                              datadir + '/t10k-labels-idx1-ubyte.gz')


def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with gfile.Open(filename, 'r') as f, gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                       (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


def extract_labels(filename):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting', filename)
  with gfile.Open(filename, 'r') as f, gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                       (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = np.frombuffer(buf, dtype=np.uint8)
    return labels


def mnist_model(inputs, is_training=True, emb_size=128, l2_weight=1e-3):  # pylint: disable=unused-argument
  """Construct the image-to-embedding vector model."""

  inputs = tf.cast(inputs, tf.float32) / 255.0
  net = inputs
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      activation_fn=tf.nn.elu,
      weights_regularizer=slim.l2_regularizer(l2_weight)):
    net = slim.conv2d(net, 32, [3, 3], scope='conv1_1')
    net = slim.conv2d(net, 32, [3, 3], scope='conv1_2')
    net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 14

    net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
    net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
    net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 7

    net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
    net = slim.conv2d(net, 128, [3, 3], scope='conv3_2')
    net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 3

    net = slim.flatten(net, scope='flatten')
    emb = slim.fully_connected(net, emb_size, scope='fc1')
  return emb

default_model = mnist_model
