from __future__ import division
from __future__ import print_function

import csv
import pickle

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import data_dirs

DATADIR = data_dirs.gtsrb

NUM_LABELS = 43
IMAGE_SHAPE = [40, 40, 3]


def get_data(name):
    """Utility for convenient data loading."""
    if name in ['train', 'unlabeled']:
        return read_gtsrb_pickle(DATADIR + '/gtsrb_train.p')
    elif name == 'test':
        return read_gtsrb_pickle(DATADIR + '/gtsrb_test.p')


def read_gtsrb_pickle(filename):
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


def preprocess_gtsrb(images, roi_boxes, resize_to):
    """
    Crops images to region-of-interest boxes and applies resizing with bilinear
    interpolation.
    :param images: np.array of images
    :param roi_boxes: np.array of region-of-interest boxes of the form
           (left, upper, right, lower)
    :return:
    """
    preprocessed_images = []
    for idx, img in enumerate(images):
        pil_img = Image.fromarray(img)
        cropped_pil_img = pil_img.crop(roi_boxes[idx])
        resized_pil_img = cropped_pil_img.resize(resize_to, Image.BILINEAR)
        preprocessed_images.append(np.asarray(resized_pil_img))

    return np.asarray(preprocessed_images)


def load_and_append_image_class(prefix, gtFile, images, labels, roi_boxes):
    gtReader = csv.reader(gtFile,
                          delimiter=';')  # csv parser for annotations file
    gtReader.next()  # skip header
    # loop over all images in current annotations file
    for row in gtReader:
        images.append(
            plt.imread(prefix + row[0]))  # the 1st column is the filename
        roi_boxes.append(
            (float(row[3]), float(row[4]), float(row[5]), float(row[6])))
        labels.append(row[7])  # the 8th column is the label
    gtFile.close()


def preprocess_and_convert_gtsrb_to_pickle(rootpath, pickle_filename,
                                           type='train'):
    """
    Reads traffic sign data for German Traffic Sign Recognition Benchmark.
    When loading the test dataset, make sure to have downloaded the EXTENDED
    annotaitons including the class ids.
    :param rootpath: path to the traffic sign data,
           for example './GTSRB/Training'
    :return: list of images, list of corresponding labels
    """
    images = []  # images
    labels = []  # corresponding labels
    roi_boxes = []  # box coordinates for ROI (left, upper, right, lower)

    if type == 'train':
        # loop over all 42 classes
        for c in range(0, NUM_LABELS):
            prefix = rootpath + '/' + format(c, '05d') + '/'  # subdir for class
            gtFile = open(
                prefix + 'GT-' + format(c, '05d') + '.csv')  # annotations file
            load_and_append_image_class(prefix, gtFile, images, labels,
                                        roi_boxes)
    elif type == 'test':
        prefix = rootpath + '/'
        gtFile = open(prefix + 'GT-final_test' + '.csv')  # annotations file
        load_and_append_image_class(prefix, gtFile, images, labels, roi_boxes)
    else:
        raise ValueError(
            'The data partition type you have provided is not valid.')

    images = np.asarray(images)
    labels = np.asarray(labels)
    roi_boxes = np.asarray(roi_boxes)

    preprocessed_images = preprocess_gtsrb(images, roi_boxes,
                                           resize_to=IMAGE_SHAPE[:-1])

    pickle.dump({'images': preprocessed_images, 'labels': labels},
                open(pickle_filename, "wb"))
