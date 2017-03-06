#! /usr/bin/env python
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

Association-based semi-supervised training module.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

#import cv2
import numpy as np
import tensorflow as tf
import semisup
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.training import saver as tf_saver


FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'svhn', 'Which dataset to work on.')

flags.DEFINE_string('target_dataset', None,
                    'If specified, perform domain adaptation using dataset as source domain '
                    'and target_dataset as target domain.')

flags.DEFINE_string('target_dataset_split', 'unlabeled',
                    'Which split of the target dataset to use for domain adaptation.')

flags.DEFINE_string('architecture', 'svhn_model', 'Which dataset to work on.')

flags.DEFINE_integer('sup_per_class', 100,
                     'Number of labeled samples used per class in total. -1 = all')

flags.DEFINE_integer('unsup_samples', -1,
                     'Number of unlabeled samples used in total. -1 = all.')

flags.DEFINE_integer('sup_seed', -1,
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_per_batch', 10,
                     'Number of labeled samples per class per batch.')

flags.DEFINE_integer('unsup_batch_size', 100,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

flags.DEFINE_float('minimum_learning_rate', 1e-6,
                   'Lower bound for learning rate.')

flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.')

flags.DEFINE_float('decay_steps', 60000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_float('visit_weight', 0.0, 'Weight for visit loss.')

flags.DEFINE_string('visit_weight_envelope', None,
                    'Increase visit weight with an envelope: [None, sigmoid, linear]')

flags.DEFINE_integer('visit_weight_envelope_steps', -1,
                     'Number of steps (after delay) at which envelope saturates. -1 = follow walker loss env.')

flags.DEFINE_integer('visit_weight_envelope_delay', -1,
                     'Number of steps at which envelope starts. -1 = follow walker loss env.')

flags.DEFINE_float('walker_weight', 1.0, 'Weight for walker loss.')

flags.DEFINE_string('walker_weight_envelope', None,
                    'Increase walker weight with an envelope: [None, sigmoid, linear]')

flags.DEFINE_integer('walker_weight_envelope_steps', 100,
                     'Number of steps (after delay) at which envelope saturates.')

flags.DEFINE_integer('walker_weight_envelope_delay', 3000,
                     'Number of steps at which envelope starts.')

flags.DEFINE_float('logit_weight', 1.0, 'Weight for logit loss.')

flags.DEFINE_integer('max_steps', 100000, 'Number of training steps.')

flags.DEFINE_bool('augmentation', False,
                  'Apply data augmentation during training.')

flags.DEFINE_integer('new_size', 0,
                     'If > 0, resize image to this width/height.')

flags.DEFINE_integer('virtual_embeddings', 0,
                     'How many virtual embeddings to add.')

flags.DEFINE_string('logdir', '/tmp/semisup', 'Training log path.')

flags.DEFINE_integer('save_summaries_secs', 150,
                     'How often should summaries be saved (in seconds).')

flags.DEFINE_integer('save_interval_secs', 300,
                     'How often should checkpoints be saved (in seconds).')

flags.DEFINE_integer('log_every_n_steps', 100,
                     'Logging interval for slim training loop.')

flags.DEFINE_integer('max_checkpoints', 5,
                     'Maximum number of recent checkpoints to keep.')

flags.DEFINE_float('keep_checkpoint_every_n_hours', 5.0,
                   'How often checkpoints should be kept.')

flags.DEFINE_float('batch_norm_decay', 0.99,
                   'Batch norm decay factor (only used for STL-10 at the moment.')

flags.DEFINE_integer('remove_classes', 0,
                     'Remove this number of classes from the labeled set, '
                     'starting with highest label number.')

flags.DEFINE_string('master', '',
                    'BNS name of the TensorFlow master to use.')

flags.DEFINE_integer('ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then the parameters '
                     'are handled locally by the worker.')

flags.DEFINE_integer('task', 0,
                     'The Task ID. This value is used when training with multiple workers to '
                     'identify each worker.')

# TODO(haeusser) convert to argparse as gflags will be discontinued
# flags.DEFINE_multi_float('custom_lr_vals', None,
#                         'For custom lr schedule: lr values.')

# flags.DEFINE_multi_int('custom_lr_steps', None,
#                       'For custom lr schedule: step values.')

FLAGS.custom_lr_vals = None
FLAGS.custom_lr_steps = None


# Data augmentation routines
def tf_affine_transformation(imgs, shape, batch_size):
    """Pyfunc wrapper for randomized affine transformations."""

    def _transform(imgs, shape, batch_size):
        """Helper function."""
        h, w = shape[:2]
        c = np.float32([w, h]) / 2.0
        mat = np.random.normal(size=[batch_size, 2, 3])
        mat[:, :2, :2] = mat[:, :2, :2] * 0.15 + np.eye(2)
        mat[:, :, 2] = mat[:, :, 2] * 2 + c - mat[:, :2, :2].dot(c)
        res = []
        for mat_i, img in zip(mat, imgs):
            border = np.random.choice([cv2.BORDER_WRAP, cv2.BORDER_REFLECT])
            out_img = cv2.warpAffine(img, mat_i, (w, h), borderMode=border)
            if np.random.rand() > 0.5:
                out_img = cv2.GaussianBlur(out_img, (7, 7), -1)
            res.append(out_img)
        return [res]

    return tf.py_func(
        _transform, [imgs, shape, batch_size], [tf.float32],
        name='affine_transform')


def apply_affine_augmentation(imgs, shape):
    imgs = tf.unstack(imgs)
    batch_size = len(imgs)
    imgs = tf_affine_transformation(imgs, shape, batch_size)
    imgs = tf.squeeze(tf.stack(imgs))
    imgs.set_shape([batch_size] + shape)
    return imgs


def rotate(x, degrees):
    def _rotate(x, degrees):
        rows, cols = x.shape[:2]
        rot_m = cv2.getRotationMatrix2D((cols / 2, rows / 2), degrees, 1)
        return cv2.warpAffine(x, rot_m, (cols, rows))

    return tf.py_func(_rotate, [x, degrees], [tf.float32], name='rotate')


def apply_augmentation_merged(inputs, shape, params):
    inputs = apply_affine_augmentation(inputs, shape)
    return apply_augmentation(inputs, shape, params)


def apply_augmentation(inputs, shape, params):
    ap = params
    with tf.name_scope('augmentation'):
        images = tf.unstack(inputs)
        out_images = []
        for image in images:
            # rotation
            angle = tf.random_uniform(
                [1],
                minval=-ap['max_rotate_angle'],
                maxval=ap['max_rotate_angle'],
                dtype=tf.float32,
                seed=None,
                name='random_angle')
            image = tf.squeeze(rotate(image, angle))

            # cropping
            if ap['max_crop_percentage']:
                crop_percentage = tf.random_uniform(
                    [1],
                    minval=0,
                    maxval=ap['max_crop_percentage'],
                    dtype=tf.float32,
                    seed=None,
                    name='random_crop_percentage')

                crop_shape = 1.0 - crop_percentage
                crop_shape = tf.multiply(np.array(shape[:2], dtype=np.float32), crop_shape)
                assert crop_shape.get_shape() == 2, 'crop shape = {}'.format(crop_shape)
                x = tf.cast(crop_shape, tf.int32)
                cropped_h, cropped_w = tf.unstack(x)
                [image] = tf.random_crop(image, [cropped_h, cropped_w])
                image = tf.image.resize_nearest_neighbor([image], shape[:2])

            # color transform prep
            raise NotImplementedError, "Need to migrate to tf.image"  # TODO(haeusser)
            color_transformations = []
            color_transformations.append(
                preprocess.random_brightness_func(ap['brightness_max_delta']))
            color_transformations.append(
                preprocess.random_saturation_func(ap['saturation_lower'], ap[
                    'saturation_upper']))
            color_transformations.append(
                preprocess.random_hue_func(ap['hue_max_delta']))
            color_transformations.append(
                preprocess.random_to_gray_func(ap['gray_prob']))

            image.set_shape([1] + shape)
            image = tf.squeeze(image)

            assert image.get_shape().as_list() == shape, 'image has shape {}'.format(
                image.get_shape().as_list())

            image = preprocess.apply_transformations(image, color_transformations)
            image, _ = preprocess.flip_dim([image])
            out_images.append(image)

        return tf.stack(out_images)


def logistic_growth(current_step, target, steps):
    """Logistic envelope from zero to target value.

    This can be used to slowly increase parameters or weights over the course of
    training.

    Args:
      current_step: Current step (e.g. tf.get_global_step())
      target: Target value > 0.
      steps: Twice the number of steps after which target/2 should be reached.
    Returns:
      TF tensor holding the target value modulated by a logistic function.

    """
    assert target > 0., 'Target value must be positive.'
    alpha = 5. / steps
    current_step = tf.cast(current_step, tf.float32)
    steps = tf.cast(steps, tf.float32)
    return target * (tf.tanh(alpha * (current_step - steps / 2.)) + 1.) / 2.


def piecewise_constant(x, boundaries, values, name=None):
    """This is tf.train.piecewise_constant.

    Due to some bug, it is inaccessible.
    Remove this when the issue is resolved.

    Piecewise constant from boundaries and interval values.

    Example: use a learning rate that's 1.0 for the first 100000 steps, 0.5
      for steps 100001 to 110000, and 0.1 for any additional steps.

    ```python
    global_step = tf.Variable(0, trainable=False)
    boundaries = [100000, 110000]
    values = [1.0, 0.5, 0.1]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

    # Later, whenever we perform an optimization step, we increment global_step.
    ```

    Args:
      x: A 0-D scalar `Tensor`. Must be one of the following types: `float32`,
        `float64`, `uint8`, `int8`, `int16`, `int32`, `int64`.
      boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
        increasing entries, and with all elements having the same type as `x`.
      values: A list of `Tensor`s or float`s or `int`s that specifies the values
        for the intervals defined by `boundaries`. It should have one more element
        than `boundaries`, and all elements should have the same type.
      name: A string. Optional name of the operation. Defaults to
        'PiecewiseConstant'.

    Returns:
      A 0-D Tensor. Its value is `values[0]` when `x <= boundaries[0]`,
      `values[1]` when `x > boundaries[0]` and `x <= boundaries[1]`, ...,
      and values[-1] when `x > boundaries[-1]`.

    Raises:
      ValueError: if types of `x` and `buondaries` do not match, or types of all
          `values` do not match.
    """
    with tf.name_scope(name, 'PiecewiseConstant',
                       [x, boundaries, values, name]) as name:
        x = tf.convert_to_tensor(x)
        # Avoid explicit conversion to x's dtype. This could result in faulty
        # comparisons, for example if floats are converted to integers.
        boundaries = [tf.convert_to_tensor(b) for b in boundaries]
        for b in boundaries:
            if b.dtype != x.dtype:
                raise ValueError('Boundaries (%s) must have the same dtype as x (%s).' %
                                 (b.dtype, x.dtype))
        values = [tf.convert_to_tensor(v) for v in values]
        for v in values[1:]:
            if v.dtype != values[0].dtype:
                raise ValueError(
                    'Values must have elements all with the same dtype (%s vs %s).' %
                    (values[0].dtype, v.dtype))

        pred_fn_pairs = {}
        pred_fn_pairs[x <= boundaries[0]] = lambda: values[0]
        pred_fn_pairs[x > boundaries[-1]] = lambda: values[-1]
        for low, high, v in zip(boundaries[:-1], boundaries[1:], values[1:-1]):
            # Need to bind v here; can do this with lambda v=v: ...
            pred = (x > low) & (x <= high)
            pred_fn_pairs[pred] = lambda v=v: v

        # The default isn't needed here because our conditions are mutually
        # exclusive and exhaustive, but tf.case requires it.
        default = lambda: values[0]
        return tf.case(pred_fn_pairs, default, exclusive=True)


def apply_envelope(type, step, final_weight, final_steps, delay):
    step = tf.cast(step-delay, tf.float32)
    final_steps += delay

    if type is None:
        value = final_weight

    elif type in ['sigmoid', 'sigmoidal', 'logistic', 'log']:
        value = logistic_growth(step, final_weight, final_steps)

    elif type in ['linear', 'lin']:
        m = float(final_weight) / final_steps
        value = m * step

    else:
        raise NameError('Invalid type: ' + str(type))

    return tf.clip_by_value(value, 0., final_weight)


def main(_):
    # Load data.
    dataset_tools = getattr(semisup, FLAGS.dataset + '_tools')
    train_images, train_labels = dataset_tools.get_data('train')
    if FLAGS.target_dataset is not None:
        target_dataset_tools = getattr(semisup, FLAGS.target_dataset + '_tools')
        train_images_unlabeled, _ = target_dataset_tools.get_data(FLAGS.target_dataset_split)
    else:
        train_images_unlabeled, _ = dataset_tools.get_data('unlabeled')

    architecture = getattr(semisup.architectures, FLAGS.architecture)

    num_labels = dataset_tools.NUM_LABELS
    image_shape = dataset_tools.IMAGE_SHAPE

    # Sample labeled training subset.
    seed = FLAGS.sup_seed if FLAGS.sup_seed != -1 else None
    sup_by_label = semisup.sample_by_label(train_images, train_labels,
                                           FLAGS.sup_per_class, num_labels, seed)

    # Sample unlabeled training subset.
    if FLAGS.unsup_samples > -1:
        num_unlabeled = len(train_images_unlabeled)
        assert FLAGS.unsup_samples <= num_unlabeled, (
            'Chose more unlabeled samples ({})'
            ' than there are in the '
            'unlabeled batch ({}).'.format(FLAGS.unsup_samples, num_unlabeled))

        rng = np.random.RandomState(seed=seed)
        train_images_unlabeled = train_images_unlabeled[rng.choice(
            num_unlabeled, FLAGS.unsup_samples, False)]

    graph = tf.Graph()
    with graph.as_default():
        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks, merge_devices=True)):

            # Set up inputs.
            t_unsup_images = semisup.create_input(train_images_unlabeled, None,
                                                  FLAGS.unsup_batch_size)
            t_sup_images, t_sup_labels = semisup.create_per_class_inputs(
                sup_by_label, FLAGS.sup_per_batch)

            if FLAGS.remove_classes:
                t_sup_images = tf.slice(
                    t_sup_images, [0, 0, 0, 0],
                    [FLAGS.sup_per_batch * (num_labels - FLAGS.remove_classes)] +
                    image_shape)

            # Resize if necessary.
            if FLAGS.new_size > 0:
                new_shape = [FLAGS.new_size, FLAGS.new_size, 3]
            else:
                new_shape = None

            # Apply augmentation
            if FLAGS.augmentation:
                if hasattr(dataset_tools, 'augmentation_params'):
                    augmentation_function = partial(
                        apply_augmentation, params=dataset_tools.augmentation_params)
                else:
                    augmentation_function = apply_affine_augmentation
            else:
                augmentation_function = None

            # Create function that defines the network.
            model_function = partial(
                architecture,
                new_shape=new_shape,
                img_shape=image_shape,
                augmentation_function=augmentation_function,
                batch_norm_decay=FLAGS.batch_norm_decay)

            # Set up semisup model.
            model = semisup.SemisupModel(model_function, num_labels, image_shape)

            # Compute embeddings and logits.
            t_sup_emb = model.image_to_embedding(t_sup_images)
            t_unsup_emb = model.image_to_embedding(t_unsup_images)

            # Add virtual embeddings.
            if FLAGS.virtual_embeddings:
                t_sup_emb = tf.concat(0, [
                    t_sup_emb, semisup.create_virt_emb(FLAGS.virtual_embeddings, 128)
                ])

                if not FLAGS.remove_classes:
                    # need to add additional labels for virtual embeddings
                    t_sup_labels = tf.concat(0, [
                        t_sup_labels,
                        (num_labels + tf.range(1, FLAGS.virtual_embeddings + 1, tf.int64))
                        * tf.ones([FLAGS.virtual_embeddings], tf.int64)
                    ])

            t_sup_logit = model.embedding_to_logit(t_sup_emb)

            # Add losses.
            visit_weight_envelope_steps = FLAGS.walker_weight_envelope_steps if FLAGS.visit_weight_envelope_steps == -1 else FLAGS.visit_weight_envelope_steps
            visit_weight_envelope_delay = FLAGS.walker_weight_envelope_delay if FLAGS.visit_weight_envelope_delay == -1 else FLAGS.visit_weight_envelope_delay
            visit_weight = apply_envelope(type=FLAGS.visit_weight_envelope, step=model.step,
                                          final_weight=FLAGS.visit_weight,
                                          final_steps=visit_weight_envelope_steps, delay=visit_weight_envelope_delay)
            walker_weight = apply_envelope(type=FLAGS.walker_weight_envelope, step=model.step,
                                           final_weight=FLAGS.walker_weight,
                                           final_steps=FLAGS.walker_weight_envelope_steps, delay=FLAGS.walker_weight_envelope_delay)
            tf.summary.scalar('Weights_Visit', visit_weight)
            tf.summary.scalar('Weights_Walker', walker_weight)

            if FLAGS.unsup_samples != 0:
                model.add_semisup_loss(
                    t_sup_emb, t_unsup_emb, t_sup_labels, visit_weight=visit_weight, walker_weight=walker_weight)

            model.add_logit_loss(t_sup_logit, t_sup_labels, weight=FLAGS.logit_weight)

            # Set up learning rate schedule if necessary.
            if FLAGS.custom_lr_vals is not None and FLAGS.custom_lr_steps is not None:
                boundaries = [
                    tf.convert_to_tensor(x, tf.int64) for x in FLAGS.custom_lr_steps
                    ]

                t_learning_rate = piecewise_constant(model.step, boundaries,
                                                     FLAGS.custom_lr_vals)
            else:
                t_learning_rate = tf.maximum(
                    tf.train.exponential_decay(
                        FLAGS.learning_rate,
                        model.step,
                        FLAGS.decay_steps,
                        FLAGS.decay_factor,
                        staircase=True),
                    FLAGS.minimum_learning_rate)

            # Create training operation and start the actual training loop.
            train_op = model.create_train_op(t_learning_rate)

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            # config.log_device_placement = True

            saver = tf_saver.Saver(max_to_keep=FLAGS.max_checkpoints,
                                   keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours)

            slim.learning.train(
                train_op,
                logdir=FLAGS.logdir + '/train',
                save_summaries_secs=FLAGS.save_summaries_secs,
                save_interval_secs=FLAGS.save_interval_secs,
                master=FLAGS.master,
                is_chief=(FLAGS.task == 0),
                startup_delay_steps=(FLAGS.task * 20),
                log_every_n_steps=FLAGS.log_every_n_steps,
                session_config=config,
                trace_every_n_steps=1000,
                saver=saver,
                number_of_steps=FLAGS.max_steps,
            )


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run()
