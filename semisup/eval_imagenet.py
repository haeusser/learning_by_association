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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import sys

import tensorflow as tf
import semisup
import tensorflow.contrib.slim as slim


from tensorflow.python.platform import app
from tensorflow.python.platform import flags

sys.path.insert(0, '/usr/wiss/haeusser/libs/tfmodels/inception')  #TODO(haeusser) use slim
from inception.imagenet_data import ImagenetData
from inception import image_processing

FLAGS = flags.FLAGS

flags.DEFINE_string('architecture', 'inception_model', 'Which dataset to work on.')

flags.DEFINE_integer('eval_batch_size', 500, 'Batch size for eval loop.')

flags.DEFINE_integer('new_size', 0, 'If > 0, resize image to this width/height.'
                     'Needs to match size used for training.')

flags.DEFINE_integer('eval_interval_secs', 300,
                     'How many seconds between executions of the eval loop.')

flags.DEFINE_string('logdir', '/tmp/semisup/imagenet',
                    'Where the checkpoints are stored '
                    'and eval events will be written to.')

flags.DEFINE_string('master', '',
                    'BNS name of the TensorFlow master to use.')


def main(_):
  # Load dataset
  tf.app.flags.FLAGS.data_dir ='/work/haeusser/data/imagenet/shards'
  dataset = ImagenetData(subset='validation')
  assert dataset.data_files()

  num_labels = dataset.num_classes() + 1
  image_shape = [FLAGS.image_size, FLAGS.image_size, 3]


  graph = tf.Graph()
  with graph.as_default():

    images, labels = image_processing.batch_inputs(
        dataset, 32, train=True,
        num_preprocess_threads=16,
        num_readers=FLAGS.num_readers)

    # Set up semisup model.
    model = semisup.SemisupModel(semisup.architectures.inception_model, num_labels, image_shape, test_in=images)

    # Add moving average variables.
    for var in tf.get_collection('moving_vars'):
      tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)
    for var in slim.get_model_variables():
      tf.add_to_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES, var)

    # Get prediction tensor from semisup model.
    predictions = tf.argmax(model.test_logit, 1)

    # Accuracy metric for summaries.
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
    })
    for name, value in names_to_values.iteritems():
      tf.summary.scalar(name, value)

    # Run the actual evaluation loop.
    num_batches = math.ceil(dataset.num_examples_per_epoch() / float(FLAGS.eval_batch_size))
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.logdir,
        logdir=FLAGS.logdir,
        num_evals=num_batches,
        eval_op=names_to_updates.values(),
        eval_interval_secs=FLAGS.eval_interval_secs,
        session_config=config)



if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run()
