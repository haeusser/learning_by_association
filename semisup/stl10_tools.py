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

Definitions and utilities for the STL10 model.

This file contains functions that are needed for semisup training and
evalutaion on the STL10 dataset.
They are used in stl10_train.py and stl10_eval.py.

"""
from __future__ import division
from __future__ import print_function

import numpy as np
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.platform import gfile
import data_dirs

DATADIR = data_dirs.stl10
NUM_LABELS = 10
IMAGE_SHAPE = [96, 96, 3]


def get_data(name, max_num=20000):
  """Utility for convenient data loading.

  Args:
    name: Name of the split. Can be 'test', 'train' or 'unlabeled'.
    max_num: maximum number of unlabeled samples.
  Returns:
    A tuple containing (images, labels) where lables=None for the unlabeled
    split.
  """
  if name == 'train':
    return extract_images(DATADIR + 'train_X.bin',
                          IMAGE_SHAPE), extract_labels(DATADIR + 'train_y.bin')
  elif name == 'test':
    return extract_images(DATADIR + 'test_X.bin',
                          IMAGE_SHAPE), extract_labels(DATADIR + 'test_y.bin')

  elif name == 'unlabeled':
    res = extract_images(DATADIR + 'unlabeled_X.bin', IMAGE_SHAPE)
    num_images = len(res)
    if num_images > max_num:
      rng = np.random.RandomState()
      return res[rng.choice(len(res), max_num, False)], None
    else:
      return res, None


def extract_images(filename, shape):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  logging.info('Extracting %s', filename)
  with gfile.Open(filename) as f:
    imgs = np.fromstring(f.read(), np.uint8)
  imgs = imgs.reshape(-1, *shape[::-1])
  imgs = np.transpose(imgs, [0, 3, 2, 1])
  return imgs


def extract_labels(filename):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  logging.info('Extracting %s', filename)
  with gfile.Open(filename) as f:
    lbls = np.fromstring(f.read(), np.uint8)
  lbls -= 1  # STL-10 labels are not zero-indexed
  return lbls


def pick_fold(images, labels, fold=-1):
  """Choose subset of labeled training data.

     According to the training protocol suggested by the creators of the dataset
     https://cs.stanford.edu/~acoates/stl10/

  Args:
    images: A 4D numpy array containing the images.
    labels: A 1D numpy array containing the corresponding labels.
    fold: The fold index in [0, 9]. Default: -1 = use all data.
  Returns:
    A tuple (images, lables)
  """
  assert -1 <= fold <= 9, 'Fold index needs to be in [0, 9] or -1 for all data.'
  if fold > -1:
    logging.info('Selecting fold %d', fold)
    fold_indices = []
    with gfile.Open(
        'path_to_stl10_binary/fold_indices.txt',
        'r') as f:
      for line in f.iteritems():
        fold_indices.append((line.split(' ')[:-1]))
    fold_indices = np.array(fold_indices).astype(np.uint16)

    images = images[fold_indices[fold]]
    labels = labels[fold_indices[fold]]
  else:
    logging.info('Using all folds.')
  return images, labels


# Dataset specific augmentation parameters.
augmentation_params = dict()
augmentation_params['max_crop_percentage'] = 0.2
augmentation_params['brightness_max_delta'] = 1.3
augmentation_params['saturation_lower'] = 0.5
augmentation_params['saturation_upper'] = 1.2
augmentation_params['hue_max_delta'] = 0.1
augmentation_params['gray_prob'] = 0.5
augmentation_params['max_rotate_angle'] = 10