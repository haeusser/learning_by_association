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


def mnist_model(inputs,
                is_training=True,
                emb_size=128,
                l2_weight=1e-3,
                batch_norm_decay=None,
                img_shape=None,
                new_shape=None,
                augmentation_function=None,
                image_summary=False):  # pylint: disable=unused-argument

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
                    augmentation_function=None,
                    img_shape=None,
                    new_shape=None,
                    batch_norm_decay=None,
                    dropout_keep_prob=0.8,
                    min_depth=16,
                    depth_multiplier=1.0,
                    spatial_squeeze=True,
                    reuse=None,
                    scope='InceptionV3',
                    num_classes=10,
                    **kwargs):
    from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
    from tensorflow.python.ops import variable_scope
    from tensorflow.contrib.framework.python.ops import arg_scope
    from tensorflow.contrib.layers.python.layers import layers as layers_lib

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
    mean = tf.reduce_mean(net, [1, 2], True)
    std = tf.reduce_mean(tf.square(net - mean), [1, 2], True)
    net = (net - mean) / (std + 1e-5)

    inputs = net

    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')

    with variable_scope.variable_scope(
            scope, 'InceptionV3', [inputs, num_classes], reuse=reuse) as scope:
        with arg_scope(
                [layers_lib.batch_norm, layers_lib.dropout], is_training=is_training):
            _, end_points = inception_v3_base(
                inputs,
                scope=scope,
                min_depth=min_depth,
                depth_multiplier=depth_multiplier,
                final_endpoint=end_point)

    net = end_points[end_point]
    net = slim.flatten(net, scope='flatten')
    with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
        emb = slim.fully_connected(net, emb_size, scope='fc')
    return emb


def inception_model_small(inputs,
                          emb_size=128,
                          is_training=True,
                          **kwargs):
    return inception_model(inputs=inputs, emb_size=emb_size, is_training=is_training,
                           end_point='Mixed_5d', **kwargs)


def vgg16_model(inputs, emb_size=128, is_training=True, img_shape=None, new_shape=None, dropout_keep_prob=0.5,
                l2_weight=0.0005,
                end_point=None, **kwargs):
    inputs = tf.cast(inputs, tf.float32)
    if new_shape is not None:
        shape = new_shape
        inputs = tf.image.resize_images(
            inputs,
            tf.constant(new_shape[:2]),
            method=tf.image.ResizeMethod.BILINEAR)
    else:
        shape = img_shape

    net = inputs
    mean = tf.reduce_mean(net, [1, 2], True)
    std = tf.reduce_mean(tf.square(net - mean), [1, 2], True)
    net = (net - mean) / (std + 1e-5)
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(l2_weight)):
        with slim.arg_scope([slim.dropout], is_training=is_training):
            net = slim.repeat(net, 2, slim.conv2d, 64, [3, 3], scope='conv1')  # 100
            net = slim.max_pool2d(net, [2, 2], scope='pool1')  # 50
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')  # 25
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')  # 12
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')  # 6
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')  # 3
            net = slim.flatten(net, scope='flatten')

            with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
                net = slim.fully_connected(net, 4096, [7, 7], activation_fn=tf.nn.relu, scope='fc6')
                if end_point == 'fc6':
                    return net
                net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                   scope='dropout6')
                emb = slim.fully_connected(net, emb_size, [1, 1], activation_fn=None, scope='fc7')

    return emb


def vgg16_model_small(inputs, emb_size=128, is_training=True, img_shape=None, new_shape=None, dropout_keep_prob=0.5,
                      **kwargs):
    return vgg16_model(inputs, emb_size, is_training, img_shape, new_shape, dropout_keep_prob, end_point='fc6',
                       **kwargs)


def alexnet_model(inputs,
                  is_training=True,
                  augmentation_function=None,
                  emb_size=128,
                  l2_weight=1e-4,
                  img_shape=None,
                  new_shape=None,
                  image_summary=False,
                  batch_norm_decay=0.99):
    """Mostly identical to slim.nets.alexnt, except for the reverted fc layers"""

    from tensorflow.contrib import layers
    from tensorflow.contrib.framework.python.ops import arg_scope
    from tensorflow.contrib.layers.python.layers import layers as layers_lib
    from tensorflow.contrib.layers.python.layers import regularizers
    from tensorflow.python.ops import init_ops
    from tensorflow.python.ops import nn_ops
    from tensorflow.python.ops import variable_scope

    trunc_normal = lambda stddev: init_ops.truncated_normal_initializer(0.0, stddev)

    def alexnet_v2_arg_scope(weight_decay=0.0005):
        with arg_scope(
                [layers.conv2d, layers_lib.fully_connected],
                activation_fn=nn_ops.relu,
                biases_initializer=init_ops.constant_initializer(0.1),
                weights_regularizer=regularizers.l2_regularizer(weight_decay)):
            with arg_scope([layers.conv2d], padding='SAME'):
                with arg_scope([layers_lib.max_pool2d], padding='VALID') as arg_sc:
                    return arg_sc

    def alexnet_v2(inputs,
                   is_training=True,
                   emb_size=4096,
                   dropout_keep_prob=0.5,
                   scope='alexnet_v2'):

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
        inputs = net

        with variable_scope.variable_scope(scope, 'alexnet_v2', [inputs]) as sc:
            end_points_collection = sc.original_name_scope + '_end_points'

            # Collect outputs for conv2d, fully_connected and max_pool2d.
            with arg_scope(
                    [layers.conv2d, layers_lib.fully_connected, layers_lib.max_pool2d],
                    outputs_collections=[end_points_collection]):
                net = layers.conv2d(
                    inputs, 64, [11, 11], 4, padding='VALID', scope='conv1')
                net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool1')
                net = layers.conv2d(net, 192, [5, 5], scope='conv2')
                net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool2')
                net = layers.conv2d(net, 384, [3, 3], scope='conv3')
                net = layers.conv2d(net, 384, [3, 3], scope='conv4')
                net = layers.conv2d(net, 256, [3, 3], scope='conv5')
                net = layers_lib.max_pool2d(net, [3, 3], 2, scope='pool5')

                net = slim.flatten(net, scope='flatten')

                # Use conv2d instead of fully_connected layers.
                with arg_scope(
                        [slim.fully_connected],
                        weights_initializer=trunc_normal(0.005),
                        biases_initializer=init_ops.constant_initializer(0.1)):
                    net = layers.fully_connected(net, 4096, scope='fc6')
                    net = layers_lib.dropout(
                        net, dropout_keep_prob, is_training=is_training, scope='dropout6')
                    net = layers.fully_connected(net, emb_size, scope='fc7')

        return net

    with slim.arg_scope(alexnet_v2_arg_scope()):
        return alexnet_v2(inputs, is_training, emb_size)
