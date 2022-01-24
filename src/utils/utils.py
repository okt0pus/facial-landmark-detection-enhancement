
import numpy as np
from matplotlib import pyplot as plt
import cv2
from multiprocessing.pool import Pool
from functools import partial


def plot_histogram(src):
    hist, bins = np.histogram(src.flatten(), 256, [0, 256])

    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()

    plt.plot(cdf_normalized, color='b')
    plt.hist(src.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()


def get_file_data(filepath: str) -> list:
    info_name = filepath.split('/')[-1].split('.')[0]
    data = cv2.imread(filepath)
    return [info_name, data]


def process_files(dir_path: str, files: list, max_workers: int = 4) -> list:
    return do_parallel([dir_path + '/' + f for f in files], partial(get_file_data), max_workers)


def do_parallel(data: list, func, max_workers: int = 4) -> list:
    pool = Pool(max_workers)
    results = pool.map(func, data)
    pool.close()
    pool.join()
    return results


def list_to_dict(lst: list) -> dict:
    result = {item[0]: item[1] for item in lst}
    return result


def filter_by_extension(listdir: list, ext: str) -> list:
    return [x for x in listdir if x.endswith(ext)]
