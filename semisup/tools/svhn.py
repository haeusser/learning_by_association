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
import numpy as np
import scipy.io
import data_dirs


DATADIR = data_dirs.svhn
NUM_LABELS = 10
IMAGE_SHAPE = [32, 32, 3]


def get_data(name):
  """Get a split from the dataset.

  Args:
   name: 'train' or 'test'

  Returns:
   images, labels
  """

  if name == 'train' or name == 'unlabeled':
    data = scipy.io.loadmat(DATADIR + 'train_32x32.mat')
  elif name == 'test':
    data = scipy.io.loadmat(DATADIR + 'test_32x32.mat')

  images = np.rollaxis(data['X'], -1)
  labels = data['y'].ravel() % 10

  if name == 'unlabeled':
    return images, None
  else:
    return images, labels