#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : model.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/10/2 下午3:55
# @ Software   : PyCharm
#-------------------------------------------------------

import tensorflow as tf

from utils.layer_utils import ConvConcat


class Generator(tf.keras.Model):
    def __init__(self, y_dim):
        super(Generator, self).__init__()

        # reshape layer => (-1,  1, 1, 10)
        self.reshape1 = tf.keras.layers.Reshape((1, 1, y_dim))

        # concat layer ((-1 100), (-1, 10)) => (-1, 110)
        self.concat1 = tf.keras.layers.Concatenate(axis=1)

        # full connect layer  => (-1, 1024)
        self.fc1 = tf.keras.layers.Dense(units=1024, use_bias=True, input_dim=(100 + 10,))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.leak_relu1 = tf.keras.layers.LeakyReLU()

        # concat layer ((-1 1024), (-1, 10)) => (-1, 1034)
        self.concat2 = tf.keras.layers.Concatenate(axis=1)

        # full connect layer (-1, 7*7*2*64)
        self.fc2 = tf.keras.layers.Dense(units=7*7*2*64, use_bias=True, input_dim=(1024))
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.leak_relu2 = tf.keras.layers.LeakyReLU()

        # reshape layer
        self.reshape2 = tf.keras.layers.Reshape((7, 7, 2*64))

        # conv concat ((-1, 7, 7, 2*64), (-1, 1, 1, 10)) => ((-1, 7, 7, 2*64 + 10)
        self.conv_concat1 = ConvConcat(axis=3)

        # transpose convd (-1, 7, 7, 138) => (-1, 7, 7, 64*2)
        self.conv_trans1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(1, 1),
                                                           padding='same', use_bias=True)
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.leak_relu3 = tf.keras.layers.LeakyReLU()

        # conv concat ((-1, 7, 7, 128), (-1, 1, 1, 10)) => ((-1, 7, 7, 128 + 10)
        self.conv_concat2 = ConvConcat(axis=3)

        # transpose convd (-1, 7, 7, 128) => (-1, 14, 14, 64)
        self.conv_trans2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2),
                                                           padding='same', use_bias=True)
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.leak_relu4 = tf.keras.layers.LeakyReLU()


        # convd(-1, 7, 7, 64) = > (-1, 14, 14, 64+10)
        self.conv_concat3 = ConvConcat(axis=3)

        # transpose convd (-1, 14, 14, 64+10) => (-1, 28, 28, 1)
        self.conv_trans3 = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(2, 2),
                                                           padding='same', use_bias=True, activation='sigmoid')

    def call(self, z, y):
        """

        :param noise: (batch_size, z_dim)
        :param label: (batch_size, y_dim)
        :return:
        """
        # (-1, 1, 1, 10)
        expend_y = self.reshape1(y)
        # concat (-1, 100+10)
        z = self.concat1((z, y))

        # (-1, 1024)
        model = self.fc1(z)
        model = self.bn1(model)
        model = self.leak_relu1(model)

        # (-1 1034)
        model = self.concat2((model, y))

        # (-1, 7*7*2*64)
        model = self.fc2(model)
        model = self.bn2(model)
        model = self.leak_relu2(model)

        # reshape (-1, 7, 7, 2*64)
        model = self.reshape2(model)

        # conv concat (-1, 7, 7, 2*64+10)
        model = self.conv_concat1((model, expend_y))

        # transpose convolution block  (-1, 7, 7, 128)
        model = self.conv_trans1(model)
        model = self.bn3(model)
        model = self.leak_relu3(model)

        # conv concat (-1, 7, 7, 128+10)
        model = self.conv_concat2((model, expend_y))

        # transpose convolution block (-1, 14, 14, 64)
        model = self.conv_trans2(model)
        model = self.bn4(model)
        model = self.leak_relu4(model)

        # (-1, 14, 14, 64+10)
        # conv concat (-1, 14, 14, 64+10)
        model = self.conv_concat3((model, expend_y))

        # transpose convolution block (-1, 28, 28, 1)
        model = self.conv_trans3(model)

        return model



class Discriminator(tf.keras.Model):
    def __init__(self, y_dim, dropout_rate=0.0):
        super(Discriminator, self).__init__()

        # reshape layer => (-1,  1, 1, 10)
        self.reshape1 = tf.keras.layers.Reshape((1, 1, y_dim))

        # conv concat ((-1, 28, 28, 1), (-1, 1, 1, 10)) => ((-1, 28, 28, 1+10)
        self.conv_concat1 = ConvConcat(axis=3)

        # (-1, 28, 28, 11) => (-1, 14, 14, 10)
        self.conv1 = tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.leak_relu1 = tf.keras.layers.LeakyReLU()
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout_rate)

        # conv concat ((-1, 14, 14, 10), (-1, 1, 1, 10)) => ((-1, 14, 14, 10+10)
        self.conv_concat2 = ConvConcat(axis=3)

        # (-1, 14, 14, 20) => (-1, 7, 7, 64)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.leak_relu2 = tf.keras.layers.LeakyReLU()
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout_rate)

        # flatten layer (-1, 7, 7, 64) => (-1, 7*7*64)
        self.flatten1 = tf.keras.layers.Flatten(data_format='channels_last')

        # concat ((-1, 7*7*64), (-1, 10)) => (-1, 7*7*64+10)
        self.concat1 = tf.keras.layers.Concatenate(axis=-1)

        # (-1, 7*7*64+10) => (-1, 1024)
        self.fc1  = tf.keras.layers.Dense(units=1024, use_bias=True, input_dim=(7*7*74,))
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.leak_relu3 = tf.keras.layers.LeakyReLU()
        self.dropout3 = tf.keras.layers.Dropout(rate=dropout_rate)

        # ((-1, 1024), (-1, 10)) => (-1, 1024+10)
        self.concat2 = tf.keras.layers.concatenate(axis=-1)

        # (-1, 1024) => (-1, 1)
        self.fc2 = tf.keras.layers.Dense(units=1)

    def call(self, x, y):
        """

        :param image: (-1, 28, 28, 1)
        :return:
        """

        # (-1, 1, 1, 10)
        expend_y = self.reshape1(y)
        # (-1, 28, 28, 11)
        x = self.conv_concat1(x, expend_y)

        # conv block 1 (-1, 28, 28, 11) => (-1, 14, 14, 10)
        model = self.conv1(x)
        model = self.bn1(model)
        model = self.leak_relu1(model)
        model = self.dropout1(model)

        # (-1, 14, 14, 20)
        model = self.conv_concat2(model, expend_y)

        # conv block 2 (-1, 14, 14, 20) => (-1,7, 7, 64)
        model = self.conv2(model)
        model = self.bn2(model)
        model = self.leak_relu2(model)
        model = self.dropout2(model)

        # reshape (-1, 7*7*64)
        model = self.flatten1(model)

        # concat 1 (-1, 7*7*64 +10)
        model = self.concat1(model, y)

        # fc1 (-1, 1024)
        model = self.fc1(model)
        model = self.bn3(model)
        model = self.leak_relu3(model)
        model = self.dropout3(model)

        # concat 1 (-1, 1024 +10)
        model = self.concat2(model, y)

        # fc2
        model = self.fc2(model)

        return model