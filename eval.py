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

Association-based semi-supervised eval module.

This script defines the evaluation loop that works with the training loop
from train.py.
"""
from functools import partial
import importlib
import math

import tensorflow as tf
from tensorflow.contrib.semisup.python.semisup import semisup
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import app
from tensorflow.python.platform import flags


FLAGS = flags.FLAGS

flags.DEFINE_string('package', 'mnist', 'Which package/dataset to work on.')

flags.DEFINE_integer('eval_batch_size', 500, 'Batch size for eval loop.')

flags.DEFINE_integer('new_size', 0, 'If > 0, resize image to this width/height.'
                     'Needs to match size used for training.')

flags.DEFINE_integer('eval_interval_secs', 300,
                     'How many seconds between executions of the eval loop.')

flags.DEFINE_string('logdir', '/tmp/semisup',
                    'Where the checkpoints are stored '
                    'and eval events will be written to.')

flags.DEFINE_string('master', 'local',
                    'BNS name of the TensorFlow master to use.')


def main(_):
  # Get dataset-related toolbox.
  tools = importlib.import_module(FLAGS.package + '_tools')

  num_labels = tools.NUM_LABELS
  image_shape = tools.IMAGE_SHAPE

  test_images, test_labels = tools.get_data('test')

  graph = tf.Graph()
  with graph.as_default():

    # Set up input pipeline.
    image, label = tf.train.slice_input_producer([test_images, test_labels])
    images, labels = tf.train.batch(
        [image, label], batch_size=FLAGS.eval_batch_size)
    images = tf.cast(images, tf.float32)
    labels = tf.cast(labels, tf.int64)

    # Reshape if necessary.
    if FLAGS.new_size > 0:
      new_shape = [FLAGS.new_size, FLAGS.new_size, 3]
    else:
      new_shape = None

    # Create function that defines the network.
    model_function = partial(
        tools.default_model,
        is_training=False,
        new_shape=new_shape,
        img_shape=image_shape,
        augmentation_function=None,
        image_summary=False)

    # Set up semisup model.
    model = semisup.SemisupModel(
        model_function,
        num_labels,
        image_shape,
        test_in=images)

    # Add moving average variables.
    for var in tf.get_collection('moving_vars'):
      tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
    for var in slim.get_model_variables():
      tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)

    # Get prediction tensor from semisup model.
    predictions = tf.argmax(model.test_logit, 1)

    # Accuracy metric for summaries.
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.accuracy(predictions, labels),
    })
    for name, value in names_to_values.iteritems():
      slim.summaries.add_scalar_summary(
          value, name, prefix='Eval', print_summary=True)

    # Run the actual evaluation loop.
    num_batches = math.ceil(len(test_labels) / float(FLAGS.eval_batch_size))
    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.logdir,
        logdir=FLAGS.logdir,
        num_evals=num_batches,
        eval_op=names_to_updates.values(),
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
  FLAGS.alsologtostderr = 1
  app.run()
