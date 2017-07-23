"""
Download USPS dataset from
http://statweb.stanford.edu/~tibs/ElemStatLearn/data.html

Explicit links:
Training: http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.train.gz
Test:     http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/zip.test.gz
"""
from __future__ import division
from __future__ import print_function

import gzip
import numpy as np
import data_dirs

DATADIR = data_dirs.usps

NUM_LABELS = 10
IMAGE_SHAPE = [16, 16, 1]


def get_data(name):
    """Utility for convenient data loading."""
    if name in ['train', 'unlabeled']:
        return extract_images_labels(DATADIR + '/zip.train.gz')
    elif name == 'test':
        return extract_images_labels(DATADIR + '/zip.test.gz')


def extract_images_labels(filename):
    print('Extracting', filename)
    with gzip.open(filename, 'rb') as f:
        raw_data = f.read().split()
    data = np.asarray([raw_data[start:start + 257]
                       for start in range(0, len(raw_data), 257)],
                      dtype=np.float32)
    images_vec = data[:, 1:]
    images = np.expand_dims(
        np.reshape(images_vec, (images_vec.shape[0], 16, 16)), axis=3)
    labels = data[:, 0].astype(int)
    return images, labels
