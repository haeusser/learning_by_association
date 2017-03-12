from __future__ import division
from __future__ import print_function

import data_dirs
import office

DATADIR = data_dirs.office

NUM_LABELS = office.NUM_LABELS
IMAGE_SHAPE = list(office.IMAGE_SIZE) + [3]


def get_data(name):
    """Utility for convenient data loading."""
    if name in ['train', 'unlabeled']:
        return office.read_office_data(DATADIR + '/dslr/images/', 'train')
    elif name == 'test':
        return office.read_office_data(DATADIR + '/dslr/images/', 'test')
