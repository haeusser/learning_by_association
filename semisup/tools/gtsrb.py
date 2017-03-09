from __future__ import division
from __future__ import print_function

import csv
import pickle

import matplotlib.pyplot as plt
import numpy as np

import data_dirs

DATADIR = data_dirs.gtsrb

NUM_LABELS = 43
IMAGE_SHAPE = [32, 32, 3]


def get_data(name):
    """Utility for convenient data loading."""
    if name in ['train', 'unlabeled']:
        return read_gtsrb_pickle(DATADIR + '/train_prot2.p')
    elif name == 'test':
        return read_gtsrb_pickle(DATADIR + '/test_prot2.p')


def read_gtsrb_pickle(filename):
    """
    Extract images from pickle file.
    :param filename:
    :return:
    """
    with open(filename, mode='rb') as f:
        data = pickle.load(f)
    return data['features'], data['labels']


def read_traffic_signs(rootpath):
    """
    Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    :param rootpath: path to the traffic sign data, for example './GTSRB/Training'
    :return: list of images, list of corresponding labels
    """
    images = []  # images
    labels = []  # corresponding labels
    # loop over all 42 classes
    for c in range(0, 43):
        prefix = rootpath + '/' + format(c, '05d') + '/'  # subdirectory for class
        gtFile = open(prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
        gtReader = csv.reader(gtFile, delimiter=';')  # csv parser for annotations file
        gtReader.next()  # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0]))  # the 1st column is the filename
            labels.append(row[7])  # the 8th column is the label
        gtFile.close()
    return np.asarray(images), np.asarray(labels)
