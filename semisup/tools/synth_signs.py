from __future__ import division
from __future__ import print_function

import pickle
import numpy as np
from PIL import Image
import csv

import data_dirs

DATADIR = data_dirs.synth_signs

NUM_LABELS = 43
IMAGE_SHAPE = [40, 40, 3]


def get_data(name):
    """Utility for convenient data loading."""
    if name in ['train', 'unlabeled']:
        return read_synth_signs_pickle(DATADIR + '/synth_signs_train.p')
    elif name == 'test':
        return read_synth_signs_pickle(DATADIR + '/synth_signs_test.p')


def read_synth_signs_pickle(filename):
    """
    Extract images from pickle file.
    :param filename:
    :return:
    """
    with open(filename, mode='rb') as f:
        data = pickle.load(f)
    if not type(data['labels'][0]) == int:
        labels = [int(x) for x in data['labels']]
    else:
        labels = data['labels']
    return np.array(data['images']), np.array(labels)


def preprocess_and_convert_synth_signs_to_pickle(rootpath):
    # take a randomly shuffled train/test split, but always the same
    np.random.seed(314)
    train_fraction = 0.9

    images = []  # images
    labels = []  # corresponding labels

    with open(rootpath + 'train_labelling.txt', 'rt') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            filepath = rootpath + row[0]
            img = Image.open(open(filepath, 'r'))
            img = img.resize(IMAGE_SHAPE[:-1], Image.BILINEAR)
            images.append(np.asarray(img))
            labels.append(int(row[1]))

    rand_idx = range(len(images))
    np.random.shuffle(rand_idx)
    split = int(len(images) * train_fraction)
    images = np.asarray(images)
    images = images[rand_idx]
    train_img = images[:split]
    test_img = images[split:]

    labels = np.asarray(labels)
    labels = labels[rand_idx]
    train_labels = labels[:split]
    test_labels = labels[split:]

    pickle.dump({'images': train_img, 'labels': train_labels},
                open('synth_signs_train.p', "wb"))
    pickle.dump({'images': test_img, 'labels': test_labels},
                open('synth_signs_test.p', "wb"))
