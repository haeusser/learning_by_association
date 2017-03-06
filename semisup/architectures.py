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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from functools import partial


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
        tf.summary.image('Inputs', inputs, max_outputs=3)

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


def dann_model(inputs,
               is_training=True,
               augmentation_function=None,
               emb_size=2048,
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
        tf.summary.image('Inputs', inputs, max_outputs=3)

    net = inputs
    mean = tf.reduce_mean(net, [1, 2], True)
    std = tf.reduce_mean(tf.square(net - mean), [1, 2], True)
    net = (net - mean) / (std + 1e-5)
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            activation_fn=tf.nn.relu,
            weights_regularizer=slim.l2_regularizer(l2_weight)):
        with slim.arg_scope([slim.dropout], is_training=is_training):
            # TODO(tfrerix) ab hier
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


def stl10_model(inputs,
                is_training=True,
                augmentation_function=None,
                emb_size=128,
                img_shape=None,
                new_shape=None,
                image_summary=False,
                batch_norm_decay=0.99):
    """Construct the image-to-embedding model."""
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
        tf.summary.image('Inputs', inputs, max_outputs=3)
    net = inputs
    net = (net - 128.0) / 128.0
    with slim.arg_scope([slim.dropout], is_training=is_training):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                normalizer_fn=slim.batch_norm,
                normalizer_params={
                    'is_training': is_training,
                    'decay': batch_norm_decay
                },
                activation_fn=tf.nn.elu,
                weights_regularizer=slim.l2_regularizer(5e-3), ):
            with slim.arg_scope([slim.conv2d], padding='SAME'):
                with slim.arg_scope([slim.dropout], is_training=is_training):
                    net = slim.conv2d(net, 32, [3, 3], scope='conv_s2')  #
                    net = slim.conv2d(net, 64, [3, 3], stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')  #
                    net = slim.conv2d(net, 64, [3, 3], scope='conv2')
                    net = slim.conv2d(net, 128, [3, 3], scope='conv2_2')
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool2')  #
                    net = slim.conv2d(net, 128, [3, 3], scope='conv3_1')
                    net = slim.conv2d(net, 256, [3, 3], scope='conv3_2')
                    net = slim.max_pool2d(net, [2, 2], stride=2, scope='pool3')  #
                    net = slim.conv2d(net, 128, [3, 3], scope='conv4')
                    net = slim.flatten(net, scope='flatten')

                    with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
                        emb = slim.fully_connected(
                            net, emb_size, activation_fn=None, scope='fc1')
    return emb


def mnist_model(inputs, is_training=True, emb_size=128, l2_weight=1e-3, batch_norm_decay=None, img_shape=None,
                augmentation_function=None, new_shape=None):  # pylint: disable=unused-argument
    """Construct the image-to-embedding vector model."""

    inputs = tf.cast(inputs, tf.float32) / 255.0
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
            inputs,
            tf.constant(new_shape[:2]),
            method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape
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


def inception_model(inputs,
                    emb_size=128,
                    is_training=True,
                    end_point='Mixed_7c',
                    **kwargs):
    from tensorflow.contrib.slim.python.slim.nets import inception_v3
    _, end_points = inception_v3.inception_v3(inputs, is_training=is_training, reuse=True, **kwargs)
    net = end_points[end_point]
    net = slim.flatten(net, scope='flatten')
    with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
        emb = slim.fully_connected(net, emb_size, scope='fc')
    return emb


def inception_model_small(inputs,
                          emb_size=128,
                          is_training=True,
                          **kwargs):
    return partial(inception_model, inputs=inputs, emb_size=emb_size, is_training=is_training, num_classes=10,
                   end_point='Mixed_5d', **kwargs)

def inception_model_medium(inputs,
                          emb_size=128,
                          is_training=True,
                          **kwargs):
    return partial(inception_model, inputs=inputs, emb_size=emb_size, is_training=is_training, num_classes=10,
                   end_point='Mixed_6e', **kwargs)