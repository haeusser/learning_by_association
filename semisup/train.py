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

import sys
from functools import partial
from importlib import import_module

import numpy as np
import semisup
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tensorflow.python.training import saver as tf_saver

FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'svhn', 'Which dataset to work on.')

flags.DEFINE_string('target_dataset', None,
                    'If specified, perform domain adaptation using dataset as '
                    'source domain and target_dataset as target domain.')

flags.DEFINE_string('target_dataset_split', 'unlabeled',
                    'Which split of the target dataset to use for domain '
                    'adaptation.')

flags.DEFINE_string('architecture', 'svhn_model', 'Which network architecture '
                    'from architectures.py to use.')

flags.DEFINE_integer('sup_per_class', 100,
                     'Number of labeled samples used per class in total.'
                     ' -1 = all')

flags.DEFINE_integer('unsup_samples', -1,
                     'Number of unlabeled samples used in total. -1 = all.')

flags.DEFINE_integer('sup_seed', -1,
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_per_batch', 10,
                     'Number of labeled samples per class per batch.')

flags.DEFINE_integer('unsup_batch_size', 100,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('emb_size', 128,
                     'Size of the embeddings to learn.')

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
                     'Number of steps (after delay) at which envelope '
                     'saturates. -1 = follow walker loss env.')

flags.DEFINE_integer('visit_weight_envelope_delay', -1,
                     'Number of steps at which envelope starts. -1 = follow '
                     'walker loss env.')

flags.DEFINE_float('walker_weight', 1.0, 'Weight for walker loss.')

flags.DEFINE_string('walker_weight_envelope', None,
                    'Increase walker weight with an envelope: [None, sigmoid, linear]')

flags.DEFINE_integer('walker_weight_envelope_steps', 100,
                     'Number of steps (after delay) at which envelope '
                     'saturates.')

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
                   'Batch norm decay factor '
                   '(only used for STL-10 at the moment.')

flags.DEFINE_integer('remove_classes', 0,
                     'Remove this number of classes from the labeled set, '
                     'starting with highest label number.')

flags.DEFINE_string('master', '',
                    'BNS name of the TensorFlow master to use.')

flags.DEFINE_integer('ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, '
                     'then the parameters '
                     'are handled locally by the worker.')

flags.DEFINE_integer('task', 0,
                     'The Task ID. This value is used when training with '
                     'multiple workers to identify each worker.')


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


def apply_envelope(type, step, final_weight, growing_steps, delay):
    assert growing_steps > 0, "Growing steps for envelope must be > 0."
    step = tf.cast(step - delay, tf.float32)
    final_step = growing_steps + delay

    if type is None:
        value = final_weight

    elif type in ['sigmoid', 'sigmoidal', 'logistic', 'log']:
        value = logistic_growth(step, final_weight, final_step)

    elif type in ['linear', 'lin']:
        m = float(final_weight) / (
            growing_steps) if not growing_steps == 0.0 else 999.
        value = m * step

    else:
        raise NameError('Invalid type: ' + str(type))

    return tf.clip_by_value(value, 0., final_weight)


def main(argv):
    del argv

    # Load data.
    dataset_tools = import_module('tools.' + FLAGS.dataset)
    train_images, train_labels = dataset_tools.get_data('train')
    if FLAGS.target_dataset is not None:
        target_dataset_tools = import_module('tools.' + FLAGS.target_dataset)
        train_images_unlabeled, _ = target_dataset_tools.get_data(
            FLAGS.target_dataset_split)
    else:
        train_images_unlabeled, _ = dataset_tools.get_data('unlabeled')

    architecture = getattr(semisup.architectures, FLAGS.architecture)

    num_labels = dataset_tools.NUM_LABELS
    image_shape = dataset_tools.IMAGE_SHAPE

    # Sample labeled training subset.
    seed = FLAGS.sup_seed if FLAGS.sup_seed != -1 else None
    sup_by_label = semisup.sample_by_label(train_images, train_labels,
                                           FLAGS.sup_per_class, num_labels,
                                           seed)

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
        with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks,
                                                      merge_devices=True)):

            # Set up inputs.
            t_unsup_images = semisup.create_input(train_images_unlabeled, None,
                                                  FLAGS.unsup_batch_size)
            t_sup_images, t_sup_labels = semisup.create_per_class_inputs(
                sup_by_label, FLAGS.sup_per_batch)

            if FLAGS.remove_classes:
                t_sup_images = tf.slice(
                    t_sup_images, [0, 0, 0, 0],
                    [FLAGS.sup_per_batch * (
                        num_labels - FLAGS.remove_classes)] +
                    image_shape)

            # Resize if necessary.
            if FLAGS.new_size > 0:
                new_shape = [FLAGS.new_size, FLAGS.new_size, image_shape[-1]]
            else:
                new_shape = None

            # Apply augmentation
            if FLAGS.augmentation:
                # TODO(haeusser) generalize augmentation
                def _random_invert(inputs, _):
                    randu = tf.random_uniform(
                        shape=[FLAGS.sup_per_batch * num_labels], minval=0.,
                        maxval=1.,
                        dtype=tf.float32)
                    randu = tf.cast(tf.less(randu, 0.5), tf.float32)
                    randu = tf.expand_dims(randu, 1)
                    randu = tf.expand_dims(randu, 1)
                    randu = tf.expand_dims(randu, 1)
                    inputs = tf.cast(inputs, tf.float32)
                    return tf.abs(inputs - 255 * randu)

                augmentation_function = _random_invert
            else:
                augmentation_function = None

            # Create function that defines the network.
            model_function = partial(
                architecture,
                new_shape=new_shape,
                img_shape=image_shape,
                augmentation_function=augmentation_function,
                batch_norm_decay=FLAGS.batch_norm_decay,
                emb_size=FLAGS.emb_size)

            # Set up semisup model.
            model = semisup.SemisupModel(model_function, num_labels,
                                         image_shape)

            # Compute embeddings and logits.
            t_sup_emb = model.image_to_embedding(t_sup_images)
            t_unsup_emb = model.image_to_embedding(t_unsup_images)

            # Add virtual embeddings.
            if FLAGS.virtual_embeddings:
                t_sup_emb = tf.concat([
                    t_sup_emb, semisup.create_virt_emb(FLAGS.virtual_embeddings,
                                                       FLAGS.emb_size)
                ], 0)

                if not FLAGS.remove_classes:
                    # need to add additional labels for virtual embeddings
                    t_sup_labels = tf.concat(0, [
                        t_sup_labels,
                        (num_labels + tf.range(1, FLAGS.virtual_embeddings + 1,
                                               tf.int64))
                        * tf.ones([FLAGS.virtual_embeddings], tf.int64)
                    ])

            t_sup_logit = model.embedding_to_logit(t_sup_emb)

            # Add losses.
            visit_weight_envelope_steps = (
                FLAGS.walker_weight_envelope_steps
                if FLAGS.visit_weight_envelope_steps == -1
                else FLAGS.visit_weight_envelope_steps)
            visit_weight_envelope_delay = (
                FLAGS.walker_weight_envelope_delay
                if FLAGS.visit_weight_envelope_delay == -1
                else FLAGS.visit_weight_envelope_delay)
            visit_weight = apply_envelope(
                            type=FLAGS.visit_weight_envelope,
                            step=model.step,
                            final_weight=FLAGS.visit_weight,
                            growing_steps=visit_weight_envelope_steps,
                            delay=visit_weight_envelope_delay)
            walker_weight = apply_envelope(
                                type=FLAGS.walker_weight_envelope,
                                step=model.step,
                                final_weight=FLAGS.walker_weight,
                                growing_steps=FLAGS.walker_weight_envelope_steps,  # pylint:disable=line-too-long
                                delay=FLAGS.walker_weight_envelope_delay)
            tf.summary.scalar('Weights_Visit', visit_weight)
            tf.summary.scalar('Weights_Walker', walker_weight)

            if FLAGS.unsup_samples != 0:
                model.add_semisup_loss(t_sup_emb,
                                       t_unsup_emb,
                                       t_sup_labels,
                                       visit_weight=visit_weight,
                                       walker_weight=walker_weight)

            model.add_logit_loss(t_sup_logit,
                                 t_sup_labels,
                                 weight=FLAGS.logit_weight)

            # Set up learning rate
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
                                   keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours)  # pylint:disable=line-too-long

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
