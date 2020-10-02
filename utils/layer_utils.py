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

    def call(self, x, y):
        x_shape = array_ops.shape(x)
        y_shape = array_ops.shape(y)

        z = tf.keras.layers.concatenate([x, y * tf.ones([x_shape[:2] + [y_shape[3]]])],
                                         axis=self.axis)

        return z


if __name__ == "__main__":
    pass