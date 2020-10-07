#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : cfgs.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/10/2 下午3:55
# @ Software   : PyCharm
#-------------------------------------------------------


from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf

# ------------------------------------------------
VERSION = 'Image_Caption_20200921'
NET_NAME = 'image_caption'


#------------------------------GPU config----------------------------------
# device_name = tf.test.gpu_device_name()
# if device_name != '/device:GPU:0':
#     raise SystemError('GPU device not found')
# print('Found GPU at: {}'.format(device_name))
# print(tf.test.is_gpu_available())
# ------------get gpu and cpu list------------------
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
# print(gpus)
# print(cpus)

# ------------------set visible of current program-------------------
# method 1 Terminal input
# $ export CUDA_VISIBLE_DEVICES = 2, 3
# method 1
# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
# method 2
tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
# ----------------------set gpu memory allocation-------------------------
# method 1: set memory size dynamic growth
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# method 2: set allocate static memory size
# tf.config.experimental.set_virtual_device_configuration(
#     device=gpus[0],
#     logical_devices = [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
# )

# ---------------------------------------- System_config----------------------------
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print (20*"++--")
print (ROOT_PATH)
GPU_GROUP = "4"
SHOW_TRAIN_INFO_INTE = 100
SMRY_ITER = 100
SAVE_WEIGHTS_ITER = 20

SUMMARY_PATH = ROOT_PATH + '/outputs/summary'
INFERENCE_SAVE_PATH = ROOT_PATH + '/outputs/inference_results'
TEST_SAVE_PATH = ROOT_PATH + '/outputs/test_results'
INFERENCE_IMAGE_PATH = ROOT_PATH + '/outputs/inference_image'
# INFERENCE_SAVE_PATH = ROOT_PATH + '/tools/inference_results'
# IMAGE_FEATURE_PATH = ROOT_PATH + '/data/image_feature'
IMAGE_SAVE_PATH = ROOT_PATH + '/outputs/generate_image'

TRAINED_CKPT = os.path.join(ROOT_PATH, 'outputs/trained_weights')
EVALUATE_DIR = ROOT_PATH + '/outputs/evaluate_result'

#----------------------Data---------------------
DATASET_PATH = ROOT_PATH + '/data'

#------------------------network config--------------------------------
Z_DIM = 100  # the dimension of noise z
Y_DIM = 10  # the dimension of condition y
OUTPUT_SIZE = 28
BATCH_SIZE = 128
# SEQUENCE_LENGTH = 100 # the number in singe time dimension of a single sequence of input data


# NUM_UNITS = [128, 64, 32]
#-------------------------train config-------------------------------
EMBEDDING_TRANSFER = False
GENERATOR_LEARNING_RATE = 1e-4
DISCRIMINATOR_LEARNING_RATE = 4e-4
NUM_EPOCH = 50
DROPOUT_RATE = 0.3

# data
SPLIT_RATIO = 0.2