import cv2
import numpy as np
from sklearn.decomposition import PCA


def gray_average(src):
    channels = cv2.split(src)
    gray = sum([c.astype(np.float64) for c in channels]) / len(channels)
    gray = gray.astype(np.uint8)
    return gray


def gray_desaturate(src):
    channels = cv2.split(src)
    c_min = np.minimum(*channels)
    c_max = np.maximum(*channels)
    gray = (c_max.astype(np.float64) + c_min.astype(np.float64)) / 2
    gray = gray.astype(np.uint8)
    return gray


def gray_decompose(src, dectype='min'):
    channels = cv2.split(src)
    c_min = np.minimum(*channels)
    c_max = np.maximum(*channels)
    gray = c_min if dectype == 'min' else c_max
    return gray


def gray_custom_shades(src, num_shades: int):
    channels = cv2.split(src)
    factor = 255.0 / (num_shades - 1)
    avg = sum([c.astype(np.float64) for c in channels]) / len(channels)
    gray = ((avg / factor) + 0.5) * factor
    gray = gray.astype(np.uint8)
    return gray


def gray_variance(src, variance):
    channels = [c.astype(np.float64) / 255 for c in cv2.split(src)]
    product = [ch * var * 255 for ch, var in zip(channels, variance)]
    gray = sum(product).astype(np.uint8)
    return gray


def pca_color2gray(image):
    orig = image

    data_input = np.concatenate(orig)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    data_input = scaler.fit_transform(data_input)

    pca = PCA(n_components=3)
    data = pca.fit_transform(data_input)

    res = gray_variance(orig, pca.explained_variance_ratio_)
    return res
