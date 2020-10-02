#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : dataset_pipeline.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/10/2 下午3:50
# @ Software   : PyCharm
#-------------------------------------------------------

import numpy as np
import os
import tensorflow  as tf

from libs.configs import cfgs


def load_mnist():
    data_dir = os.path.join(cfgs.DATASET_PATH, "mnist")
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_x = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_y = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_x = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_y = loaded[8:].reshape((10000)).astype(np.float)

    train_y = np.asarray(train_y)
    test_y = np.asarray(test_y)

    x = np.concatenate((train_x, test_x), axis=0)
    y = np.concatenate((train_y, test_y), axis=0)

    # np.random.seed(seed)
    # np.random.shuffle(X)
    # np.random.seed(seed)
    # np.random.shuffle(y)

    # convert label to one-hot
    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, int(y[i])] = 1.0

    return x, y_vec


def sample_label(num_label=10*10):
    label_vector = np.zeros((num_label , 10), dtype=np.float)
    for i in range(0 , num_label):
        label_vector[i , int(i/10)] = 1.0

    return label_vector

if __name__ == "__main__":
    load_mnist()


