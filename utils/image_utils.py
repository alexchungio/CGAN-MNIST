#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : image_utils.py
# @ Description:  
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/10/2 下午5:31
# @ Software   : PyCharm
#-------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt


def get_image(image_path , is_grayscale = False):
    return np.array(inverse_transform(imread(image_path, is_grayscale)))


def save_images(images , size , image_path):
    return imsave(inverse_transform(images) , size , image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return plt.imread(path, format='gray').astype(np.float)
    else:
        return plt.imread(path).astype(np.float)

def imsave(images , size , path):
    return plt.imsave(path , merge(images , size))

def merge(images , size):
    h , w = images.shape[1] , images.shape[2]
    img = np.zeros((h*size[0] , w*size[1] , 3))
    for idx , image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h +h , i*w : i*w+w , :] = image

    return img

def inverse_transform(image):
    return (image + 1.)/2.

def read_image_list(category):
    filenames = []
    print("list file")
    list = os.listdir(category)

    for file in list:
        filenames.append(category + "/" + file)

    print("list file ending!")

    return filenames

##from caffe
def vis_square(visu_path , data , type):
    """Take an array of shape (n, height, width) or (n, height, width , 3)
       and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))

    padding = (((0, n ** 2 - data.shape[0]) ,
                (0, 1), (0, 1))  # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data , padding, mode='constant' , constant_values=1)  # pad with ones (white)

    # tilethe filters into an im age
    data = data.reshape((n , n) + data.shape[1:]).transpose((0 , 2 , 1 , 3) + tuple(range(4 , data.ndim + 1)))

    data = data.reshape((n * data.shape[1] , n * data.shape[3]) + data.shape[4:])

    plt.imshow(data[:,:,0])
    plt.axis('off')

    if type:
        plt.savefig('./{}/weights.png'.format(visu_path) , format='png')
    else:
        plt.savefig('./{}/activation.png'.format(visu_path) , format='png')