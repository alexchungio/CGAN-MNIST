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


def generate_dataset(images, labels, batch_size, buffer_size=70000):

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))

    dataset = dataset.map(lambda item1, item2: tf.numpy_function(image_process,
                                                                 inp=[item1, item2],
                                                                 Tout=[tf.float32, tf.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size, drop_remainder=True)

    return dataset


def image_process(image, label):

    image = (image/255.).astype('float32')
    label = label.astype('float32')

    return image, label


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

    # dataset batch
    images, labels = load_mnist()
    dataset = generate_dataset(images, labels, batch_size=cfgs.BATCH_SIZE)
    for image_batch, label_batch in dataset.take(1):
        print(image_batch.shape, label_batch.shape)

    # sample label
    sample_label = sample_label()
    print(sample_label)







