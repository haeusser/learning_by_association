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

import tensorflow as tf
import sys


import semisup
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.python.slim.nets import inception_v3

from tensorflow.python.platform import app
from tensorflow.python.platform import flags

sys.path.insert(0, '/usr/wiss/haeusser/libs/tfmodels/inception')

FLAGS = flags.FLAGS


flags.DEFINE_string('architecture', 'svhn', 'Which dataset to work on.')



flags.DEFINE_integer('sup_per_class', 10,
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('sup_seed', -1,
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_per_batch', 10,
                     'Number of labeled samples per class per batch.')

flags.DEFINE_integer('unsup_batch_size', 1000,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('sup_batch_size', 1000,
                     'Number of labeled samples per batch.')

flags.DEFINE_integer('eval_interval', 500,
                     'Number of steps between evaluations.')

flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

flags.DEFINE_float('minimum_learning_rate', 3e-6, 'Final learning rate.')

flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.')

flags.DEFINE_float('decay_steps', 5000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_float('visit_weight', 1.0, 'Weight for visit loss.')

flags.DEFINE_integer('max_steps', 20000, 'Number of training steps.')

flags.DEFINE_string('logdir', '/tmp/semisup/imagenet', 'Training log path.')

flags.DEFINE_integer('save_summaries_secs', 150,
                     'How often should summaries be saved (in seconds).')

flags.DEFINE_integer('save_interval_secs', 300,
                     'How often should checkpoints be saved (in seconds).')

flags.DEFINE_integer('log_every_n_steps', 100,
                     'Logging interval for slim training loop.')

flags.DEFINE_string('master', '',
                    'BNS name of the TensorFlow master to use.')

flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

# TODO(haeusser) convert to argparse as gflags will be discontinued
#flags.DEFINE_multi_float('custom_lr_vals', None,
#                         'For custom lr schedule: lr values.')

#flags.DEFINE_multi_int('custom_lr_steps', None,
#                       'For custom lr schedule: step values.')

FLAGS.custom_lr_vals = None
FLAGS.custom_lr_steps = None
FLAGS.data_dir ='/work/haeusser/data/imagenet/shards'
FLAGS.num_readers = 16
FLAGS.input_queue_memory_factor = 16
FLAGS.image_size = 120 #299  # remember to change variable IMAGE_SHAPE


def inception_model(inputs,
                    emb_size=128,
                    is_training=True):
    _, end_points = inception_v3.inception_v3(inputs, is_training=is_training, reuse=True)
    net = end_points['Mixed_7c']
    net = slim.flatten(net, scope='flatten')
    with slim.arg_scope([slim.fully_connected], normalizer_fn=None):
        emb = slim.fully_connected(net, emb_size, scope='fc')
    return emb

def main(_):
    from inception.imagenet_data import ImagenetData
    from inception import image_processing
    dataset = ImagenetData(subset='train')
    assert dataset.data_files()
    NUM_LABELS = dataset.num_classes() + 1
    IMAGE_SHAPE = [FLAGS.image_size, FLAGS.image_size, 3]
    graph = tf.Graph()
    with graph.as_default():
        model = semisup.SemisupModel(inception_model, NUM_LABELS,
                                     IMAGE_SHAPE)

        # t_sup_images, t_sup_labels = tools.get_data('train')
        # t_unsup_images, _ = tools.get_data('unlabeled')

        images, labels = image_processing.batch_inputs(
            dataset, 32, train=True,
            num_preprocess_threads=FLAGS.num_readers,
            num_readers=FLAGS.num_readers)

        t_sup_images, t_sup_labels = tf.train.batch(
            [images, labels],
            batch_size=FLAGS.sup_batch_size,
            enqueue_many=True,
            num_threads=FLAGS.num_readers,
            capacity=1000 + 3 * FLAGS.sup_batch_size,
        )

        t_unsup_images, t_unsup_labels = tf.train.batch(
            [images, labels],
            batch_size=FLAGS.sup_batch_size,
            enqueue_many=True,
            num_threads=FLAGS.num_readers,
            capacity=1000 + 3 * FLAGS.sup_batch_size,
        )

        # Compute embeddings and logits.
        t_sup_emb = model.image_to_embedding(t_sup_images)
        t_unsup_emb = model.image_to_embedding(t_unsup_images)
        t_sup_logit = model.embedding_to_logit(t_sup_emb)

        # Add losses.
        model.add_semisup_loss(
            t_sup_emb, t_unsup_emb, t_sup_labels, visit_weight=FLAGS.visit_weight)

        model.add_logit_loss(t_sup_logit, t_sup_labels)


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

        slim.learning.train(
          train_op,
          logdir=FLAGS.logdir,
          save_summaries_secs=FLAGS.save_summaries_secs,
          save_interval_secs=FLAGS.save_interval_secs,
          master=FLAGS.master,
          is_chief=(FLAGS.task == 0),
          startup_delay_steps=(FLAGS.task * 20),
          log_every_n_steps=FLAGS.log_every_n_steps,
          session_config=config)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run()
