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
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf

import data_dirs

DATADIR = data_dirs.imagenet
LABELS_FILE = data_dirs.imagenet_labels

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
