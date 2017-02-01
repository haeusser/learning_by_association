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
import os
import numpy as np
import scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import gfile


DATADIR = '/work/haeusser/data/imagenet/raw-data/'
LABELS_FILE = '/usr/wiss/haeusser/libs/tfmodels/inception/inception/data/imagenet_lsvrc_2015_synsets.txt'

NUM_LABELS = 1000
IMAGE_SHAPE = [299, 299, 3]


def get_data(name, batch_size=1000):
    """Get a split from the dataset.

    Args:
    name: 'train' or 'test'

    Returns:
    images, labels
    """

    assert name in ['train', 'validation'], 'Allowed splits: train, validation'

    filenames, synsets, labels = _find_image_files(os.path.join(DATADIR, name), LABELS_FILE, NUM_LABELS)

    g = tf.Graph()
    min_after_dequeue = 1000
    capacity = min_after_dequeue + 3 * batch_size

    with g.as_default():
        image, label = tf.train.slice_input_producer(
                                        [filenames, labels],
                                        shuffle=False)

        image = tf.read_file(image)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_bilinear([image], IMAGE_SHAPE[:2])
        image = tf.squeeze(image)
        image.set_shape(IMAGE_SHAPE)
        image_batch, label_batch = tf.train.batch(
                                            [image, label],
                                            batch_size=batch_size
                                            )

    images = []
    labels = []

    with tf.Session(graph=g) as sess:
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            runs = len(filenames) / batch_size
            for i in xrange(runs):
                print('run %d of %d' % (i + 1, runs))
                imgs, lbls = sess.run([image_batch, label_batch])

                images.append(imgs)
                labels.append(lbls)

        except tf.errors.OutOfRangeError:
            pass
        finally:

            coord.request_stop()
            coord.join(threads)

    return images, labels



def _find_image_files(data_dir, labels_file, num_classes=1000):
  """Build a list of all images files and labels in the data set.
  Args:
    data_dir: string, path to the root directory of images.
      Assumes that the ImageNet data set resides in JPEG files located in
      the following directory structure.
        data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
        data_dir/n01440764/ILSVRC2012_val_00000543.JPEG
      where 'n01440764' is the unique synset label associated with these images.
    labels_file: string, path to the labels file.
      The list of valid labels are held in this file. Assumes that the file
      contains entries as such:
        n01440764
        n01443537
        n01484850
      where each line corresponds to a label expressed as a synset. We map
      each synset contained in the file to an integer (based on the alphabetical
      ordering) starting with the integer 1 corresponding to the synset
      contained in the first line.
      The reason we start the integer labels at 1 is to reserve label 0 as an
      unused background class.
  Returns:
    filenames: list of strings; each string is a path to an image file.
    synsets: list of strings; each string is a unique WordNet ID.
    labels: list of integer; each integer identifies the ground truth.
  """
  print('Determining list of input files and labels from %s.' % data_dir)
  challenge_synsets = [l.strip() for l in
                       tf.gfile.FastGFile(labels_file, 'r').readlines()]

  labels = []
  filenames = []
  synsets = []

  # Leave label index 0 empty as a background class.
  label_index = 1

  # Construct the list of JPEG files and labels.
  for synset in challenge_synsets[:num_classes]:
    jpeg_file_path = '%s/%s/*.JPEG' % (data_dir, synset)
    matching_files = tf.gfile.Glob(jpeg_file_path)

    labels.extend([label_index] * len(matching_files))
    synsets.extend([synset] * len(matching_files))
    filenames.extend(matching_files)

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(challenge_synsets)))
    label_index += 1

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = list(range(len(filenames)))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  synsets = [synsets[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]

  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(challenge_synsets[:num_classes]), data_dir))
  return filenames, synsets, labels



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


def inception_v3(inputs,
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
  

default_model = inception_v3
