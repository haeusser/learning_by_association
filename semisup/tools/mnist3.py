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

Definitions and utilities for the MNIST model.

This file contains functions that are needed for semisup training and evalutaion
on the MNIST dataset.
They are used in MNIST_train_eval.py.

"""

import numpy as np

import mnist

NUM_LABELS = mnist.NUM_LABELS
IMAGE_SHAPE = mnist.IMAGE_SHAPE[:2] + [3]


def get_data(name):
    """Utility for convenient data loading."""
    images, labels = mnist.get_data(name)
    images = np.concatenate([images] * 3, 3)
    return images, labels
