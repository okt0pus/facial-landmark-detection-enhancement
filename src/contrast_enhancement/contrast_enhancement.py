from enum import Enum

import numpy as np
import cv2


class CType(Enum):
    USE_COLOR = 0
    USE_LAB_L = 1


def min_max_contrast_stretching(src):
    p_min, p_max = np.min(src), np.max(src)
    f = 255 / (p_max - p_min)
    dst = f * (src - p_min)
    dst = dst.astype(np.uint8)
    return dst


def gamma_transform(src,  gamma=0.4):
    return (((src.astype(np.float64) / 255.0) ** gamma) * 255.0).astype(np.uint8)


# g_max: pre-set maximum gain of dynamic range
def weighted_threshold_histogram_equalization(src, w_out_list=np.zeros(10), r=0.5, v=0.5, g_max=1.5):
    height, width = src.shape

    hist, _ = np.histogram(src.reshape(1, -1), bins=256, range=(0, 255))

    # probability density function
    pdf = hist / (height * width)
    # probability lower/upper
    pl = 1e-4
    pu = v * np.max(pdf)
    # weighted and threshold pdf
    pwt = np.zeros_like(pdf)

    for index, pmf in enumerate(pdf):
        if pmf < pl:
            pwt[index] = 0
        elif pmf > pu:
            pwt[index] = pu
        else:
            pwt[index] = (((pmf - pl) / (pu - pl)) ** r) * pu

    # weighted and threshold cumulative distribution function
    cwt = np.cumsum(pwt)
    # normalized cwt
    cwt_n = cwt / cwt[-1]

    # dynamic range of the input image
    w_in = len(np.where(pdf > 0)[0])
    # dynamic range of the output image
    w_out = min(255.0, g_max * w_in)

    # weighted average of the current w_out value
    # and the previous ones for videos
    if np.where(w_out_list > 0)[0].size == w_out_list.size:
        w_out = (np.sum(w_out_list) + w_out) / (1 + w_out_list.size)

    # image flatten
    f = src.copy().flatten()
    f_tilde = w_out * cwt_n[f]

    # mean adjustment factor
    m_adj = np.mean(src) - np.mean(f_tilde)
    f_tilde = f_tilde + m_adj

    # adjust values
    f_tilde = np.where(f_tilde >= 0, f_tilde, np.zeros_like(f_tilde))
    f_tilde = np.where(f_tilde <= 255, f_tilde, 255 * np.ones_like(f_tilde))
    f_tilde = f_tilde.astype(np.uint8)

    return f_tilde.reshape(height, width)


def contrast_enhancement(src, eq_type=CType.USE_COLOR, eq_func=cv2.equalizeHist, eq_func_params=()):
    if eq_type == CType.USE_COLOR:
        # GRAY
        if len(src.shape) == 2:
            params = [src] + [*eq_func_params]
            return eq_func(*params)
        # BGR
        elif len(src.shape) == 3:
            b, g, r = cv2.split(src)
            params = [b] + [*eq_func_params]
            equ_b = eq_func(*params)
            params = [g] + [*eq_func_params]
            equ_g = eq_func(*params)
            params = [r] + [*eq_func_params]
            equ_r = eq_func(*params)

            return cv2.merge((equ_b, equ_g, equ_r))

    if eq_type == CType.USE_LAB_L:
        # BGR
        if len(src.shape) == 3:
            lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
            L, a, b = cv2.split(lab)

            params = [L] + [*eq_func_params]
            equ_l = eq_func(*params)

            equ_lab = cv2.merge((equ_l, a, b))
            return cv2.cvtColor(equ_lab, cv2.COLOR_LAB2BGR)

    return None

