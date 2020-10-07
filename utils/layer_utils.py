#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : layer_utils.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/10/2 下午6:56
# @ Software   : PyCharm
#-------------------------------------------------------

import tensorflow as tf
from tensorflow.python.ops import array_ops

class ConvConcat(tf.keras.layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        super(ConvConcat, self).__init__(**kwargs)
        self.axis = axis

    def build(self, axis):
        pass

    def call(self, inputs):
        x_shape = inputs[0].get_shape()
        y_shape = inputs[1].get_shape()

        z = tf.keras.layers.concatenate([inputs[0], inputs[1] * tf.ones(x_shape[:3] + [y_shape[3]])],
                                         axis=self.axis)

        return z


if __name__ == "__main__":
    pass