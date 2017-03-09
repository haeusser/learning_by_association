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

Definitions and utilities for the synthetic digit (Ganin) model.

This file contains functions that are needed for semisup training and
evalutaion on the SVHN dataset.
They are used in svhn_train.py and svhn_eval.py.

"""
from __future__ import division
from __future__ import print_function
import numpy as np
import scipy.io
import data_dirs

DATADIR = data_dirs.synth
NUM_LABELS = 10
IMAGE_SHAPE = [32, 32, 3]


def get_data(name, num=70000):
  """Get a split from the synth dataset.

  Args:
   name: 'train' or 'test'
   num: How many samples to read (randomly) from the data set

  Returns:
   images, labels
  """

  if name == 'train' or name == 'unlabeled':
    fn = 'synth_train_32x32.mat'
  elif name == 'test':
    fn = 'synth_test_32x32.mat'

  data = scipy.io.loadmat(DATADIR + fn)

  images = np.rollaxis(data['X'], -1)
  labels = data['y'].ravel() % 10

  num_samples = len(images)
  indices = np.random.choice(num_samples, min(num, num_samples), False)

  images = images[indices]
  labels = labels[indices]

  if name == 'unlabeled':
    return images, None
  else:
    return images, labels