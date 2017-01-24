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


Definitions and utilities for the svhn model.

This file contains functions that are needed for semisup training and
evalutaion on the SVHN dataset.
They are used in svhn_train.py and svhn_eval.py.
"""
import numpy as np
import scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import gfile


DATADIR = '/work/haeusser/data/svhn/'
NUM_LABELS = 10
IMAGE_SHAPE = [32, 32, 3]


def get_data(name):
  """Get a split from the dataset.

  Args:
   name: 'train' or 'test'

  Returns:
   images, labels
  """

  if name == 'train' or name == 'unlabeled':
    #  data = scipy.io.loadmat(gfile.Open(DATADIR + 'train_32x32.mat'))
    data = scipy.io.loadmat(DATADIR + 'train_32x32.mat')
  elif name == 'test':
    #  data = scipy.io.loadmat(gfile.Open(DATADIR + 'test_32x32.mat'))
    data = scipy.io.loadmat(DATADIR + 'test_32x32.mat')

  images = np.rollaxis(data['X'], -1)
  labels = data['y'].ravel() % 10

  if name == 'unlabeled':
    return images, None
  else:
    return images, labels


def svhn_model(inputs,
               is_training=True,
               augmentation_function=None,
               emb_size=128,
               l2_weight=1e-4,
               img_shape=None,
               new_shape=None,
               image_summary=False,
               batch_norm_decay=0.99):  # pylint: disable=unused-argument
  """Construct the image-to-embedding vector model."""
  inputs = tf.cast(inputs, tf.float32)
  if new_shape is not None:
    shape = new_shape
    inputs = tf.image.resize_images(
        inputs,
        tf.constant(new_shape[:2]),
        method=tf.image.ResizeMethod.BILINEAR)
  else:
    shape = img_shape
  if is_training and augmentation_function is not None:
    inputs = augmentation_function(inputs, shape)
  if image_summary:
    tf.image_summary('Inputs', inputs, max_images=3)

  net = inputs
  mean = tf.reduce_mean(net, [1, 2], True)
  std = tf.reduce_mean(tf.square(net - mean), [1, 2], True)
  net = (net - mean) / (std + 1e-5)
  with slim.arg_scope(
      [slim.conv2d, slim.fully_connected],
      activation_fn=tf.nn.elu,
      weights_regularizer=slim.l2_regularizer(l2_weight)):
    with slim.arg_scope([slim.dropout], is_training=is_training):
      net = slim.conv2d(net, 32, [3, 3], scope='conv1')
      net = slim.conv2d(net, 32, [3, 3], scope='conv1_2')
      net = slim.conv2d(net, 32, [3, 3], scope='conv1_3')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 14
      net = slim.conv2d(net, 64, [3, 3], scope='conv2_1')
      net = slim.conv2d(net, 64, [3, 3], scope='conv2_2')
      net = slim.conv2d(net, 64, [3, 3], scope='conv2_3')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 7
      net = slim.conv2d(net, 128, [3, 3], scope='conv3')
      net = slim.conv2d(net, 128, [3, 3], scope='conv3_2')
      net = slim.conv2d(net, 128, [3, 3], scope='conv3_3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 3
      net = slim.flatten(net, scope='flatten')

      with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
        emb = slim.fully_connected(net, emb_size, scope='fc1')

  return emb


default_model = svhn_model
