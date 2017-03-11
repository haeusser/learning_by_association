from __future__ import division
from __future__ import print_function

import data_dirs
from office import read_office_data

DATADIR = data_dirs.office

NUM_LABELS = 10
IMAGE_SHAPE = [128, 128, 3]


def get_data(name):
    """Utility for convenient data loading."""
    if name in ['train', 'unlabeled']:
        return read_office_data(DATADIR + '/dslr/images/', 'train')
    elif name == 'test':
        return  read_office_data(DATADIR + '/dslr/images/', 'test')


