# -*- coding: utf-8 -*-
"""
python3
description :Fingerprint image normalization
"""
import cv2
import numpy as np
import os


def bin_index(hist, percent, nf):
    sum_value = 0
    for bin_idx, bin_value in enumerate(hist):
        sum_value += bin_value
        if 100 * sum_value / nf > percent:
            return bin_idx
            # break


def lut_im(min_bin_idx, max_bin_idx):
    lut = np.zeros(256, dtype=np.uint8)
    for i, v in enumerate(lut):
        if i < min_bin_idx:
            lut[i] = 0
        elif i > max_bin_idx:
            lut[i] = 255
        else:
            lut[i] = int(255.0 * (i - min_bin_idx) / (max_bin_idx - min_bin_idx) + 0.5)
    return lut


def normalize(in_path, out_dir):
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass

    im = cv2.imread(in_path, 0)
    nf = im.size
    hist = cv2.calcHist([im], [0], None, [256], [0.0, 255.0])
    min_idx = bin_index(hist, 1, nf)
    max_idx = 255 - bin_index(reversed(hist), 1, nf)

    lut_result = lut_im(min_idx, max_idx)
    result = cv2.LUT(im, lut_result)
    prefix = os.path.splitext(os.path.basename(in_path))[0]
    img_out = out_dir + prefix + '_normalized.png'
    cv2.imwrite(img_out, result)
    return img_out



