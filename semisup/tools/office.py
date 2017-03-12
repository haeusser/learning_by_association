from __future__ import division
from __future__ import print_function

import os
import numpy as np
from PIL import Image
import random

NUM_LABELS = 31
IMAGE_SIZE = (96, 96)

def read_office_data(fileroot, partition):
    # take a randomly shuffled train/test split, but always the same
    random.seed(314)
    train_fraction = 0.7

    dirs = [f for f in os.listdir(fileroot) if not f.startswith('.')]
    dirs.sort()
    images = []
    labels = []
    for label, path in enumerate(dirs):
        filenames = os.listdir(os.path.join(fileroot, path))
        random.shuffle(filenames)
        num_samples = len(filenames)
        split = int(train_fraction * num_samples)
        if partition == 'train':
            taken_samples = filenames[:split]
        else:
            taken_samples = filenames[split:]

        for filename in taken_samples:
            filepath = os.path.join(fileroot, path, filename)
            raw_img = Image.open(filepath)
            raw_img_resized = raw_img.resize(IMAGE_SIZE)
            img = np.asarray(raw_img_resized)
            images.append(img)
            labels.append(label)

    return np.asarray(images), np.asarray(labels)