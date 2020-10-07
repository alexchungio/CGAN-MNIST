#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------------------
# @ File       : tools.py
# @ Description:
# @ Author     : Alex Chung
# @ Contact    : yonganzhong@outlook.com
# @ License    : Copyright (c) 2017-2018
# @ Time       : 2020/10/7 下午4:28
# @ Software   : PyCharm
#-------------------------------------------------------


import math
import sys
import os
import glob
import imageio


def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r{0}:[{1}{2}]{3}%\t{4}/{5}'.format (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total)
    sys.stdout.write(r)
    sys.stdout.flush()


def makedir(path):
    """
    create dir
    :param path:
    :return:
    """
    try:
        if os.path.exists(path) is False:
            os.makedirs(path)
            print('{0} has been created'.format(path))
        else:
            pass
    except Exception as e:
        print(e)


def generate_gif(image_path, anim_file):
    """

    :param image_path:
    :param anim_file:
    :return:
    """
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(os.path.join(image_path, '*.png'))
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)