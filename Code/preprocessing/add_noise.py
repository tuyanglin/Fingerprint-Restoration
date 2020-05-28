#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
description: This is used to add incomplete blocks to fingerprint images, just for our project need.
              If your image is incomplete, you don't need this script
"""
import os
import cv2
import numpy as np
import random
import math


def load_data(img_path):
    """
    load image from given path
    :param img_path:
    :return: img_raw (original image, numpy.ndarray),
             region (approximate fingerprint region, numpy.ndarray)
    """
    img_raw = cv2.imread(img_path, 0)
    bin_img = cv2.threshold(img_raw, 127, 255, type=cv2.THRESH_BINARY)[1]
    region = np.argwhere(bin_img == 0)
    region[:, [0, 1]] = region[:, [1, 0]]
    return img_raw, region


def add_img_noise(img_path, out_dir='./train_set/', shape='ellipse', mode='default', num=1):
    """
    在原图中添加噪声
    :param img_path:
    :param out_dir:
    :param shape: ellipse[default], line, rectangle
    :param mode:
    :param num:
    :return:
    """
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    img_file = os.path.basename(img_path)
    prefix = img_file.split('_normal')[0]
    suffix = str(num)

    img, region = load_data(img_path)
    # copy original image to output directory (train_set)
    # cv2.imwrite('{out_dir}{prefix}_original.png'.format_map(vars()), img)

    # get image properties
    x_range = (np.min(region[:, 0]), np.max(region[:, 0]))
    y_range = (np.min(region[:, 1]), np.max(region[:, 1]))

    # specify noise type [default: ellipse]
    if shape == 'line':
        add_line_noise(img, x_range, y_range, mode, num)
    elif shape == 'rectangle':
        add_rectangle_noise(img, x_range, y_range, mode, num)
    else:
        add_ellipse_noise(img, x_range, y_range, mode, num)

    img_out = '{out_dir}{prefix}_{mode}_noise{suffix}.png'.format_map(vars())
    cv2.imwrite(img_out, img)
    return img_out


def add_ellipse_noise(img, x_range, y_range, size='small', num=1):
    """
    Add ellipse noise to original image
    :param img:
    :param x_range:
    :param y_range:
    :param size:
    :param num:
    :return:
    """
    if size == 'large':
        coeff = 0.2
    else:  # default: size='small'
        coeff = 0.12
    a, b = map(lambda x: math.ceil(0.5 * coeff * abs(x[0] - x[1])), (x_range, y_range))
    x_lim, y_lim = map(lambda x: (x[0] + 2 * b, x[1] - 2 * b), (x_range, y_range))

    while num >= 1:
        center = (random.randint(x_lim[0], x_lim[1]), random.randint(y_lim[0], y_lim[1]))
        amorphous_ellipse(img, center, (a, b))
        num -= 1

    return img


def amorphous_ellipse(img, center, axis):
    """
    Add amorphous ellipse to original image
    :param img:
    :param center:
    :param axis:
    :return:
    """
    cnt = random.randint(2, 4)
    angle = random.sample(range(180), cnt)
    cv2.ellipse(img, center, axis, angle[0], 0, 366, 240, thickness=-1)
    cv2.ellipse(img, center, axis, angle[1], 0, 366, 240, thickness=-1)

    return img


def add_line_noise(img, x_range, y_range, line_type='default', num=1):
    """
    Add straight line noise to original image
    :param img:
    :param x_range:
    :param y_range:
    :param line_type:
    :param num:
    :return:
    """
    if line_type.find('bode') != -1:
        width = 8
    else:
        width = 3

    while num >= 1:
        x_pair, y_pair = map(lambda x: random.sample(range(x[0], x[1]), 2), (x_range, y_range))
        start = (x_pair[0], y_pair[0])
        end = (x_pair[1], y_pair[1])
        # x_pair = random.sample(range(x_range[0], x_range[1]), 2)
        # start = (x_pair[0], y_range[0])
        # end = (x_pair[1], y_range[1])
        cv2.line(img, start, end, color=255, thickness=width)
        num -= 1
    return img


def add_rectangle_noise(img, x_range, y_range, size='default', num=1):
    """
    Add rectangle noise to original image
    :param img:
    :param x_range:
    :param y_range:
    :param size:
    :param num:
    :return:
    """
    if size == 'large':
        coeff = 0.2
    else:
        coeff = 0.15
    a, b = map(lambda x: math.ceil(0.5 * coeff * abs(x[0] - x[1])), (x_range, y_range))
    x_lim, y_lim = map(lambda x: (x[0] + b, x[1] - b), (x_range, y_range))
    while num >= 1:
        center = (random.randint(x_lim[0], x_lim[1]), random.randint(y_lim[0], y_lim[1]))
        up_left = (center[0] - a, center[1] + b)
        down_right = (center[0] + a, center[1] - b)
        cv2.rectangle(img, up_left, down_right, color=255, thickness=-1)
        num -= 1
    return img


