from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
import data_dirs

DATADIR = data_dirs.mnistm


NUM_LABELS = 10
IMAGE_SHAPE = [28, 28, 3]


def get_data(name):
    """Utility for convenient data loading."""
    if name in ['train', 'unlabeled']:
        return load_mnistm(DATADIR, 'train')
    elif name == 'test':
        return load_mnistm(DATADIR, 'test')


def load_mnistm(fileroot, partition):
    with open(fileroot + 'mnistm_data.pkl', 'rb') as f:
        data = pickle.load(f)

    if partition == 'train':
        images = np.concatenate((data['train_images'],
                                 data['valid_images']), axis=0)
        labels = np.concatenate((data['train_labels'],
                                 data['valid_labels']), axis=0)
    elif partition == 'test':
        images = data['test_images']
        labels = data['test_labels']
    else:
        raise ValueError('The provided data partition name is not valid. '
                         'Use "train" or "test".')

    return images, labels

