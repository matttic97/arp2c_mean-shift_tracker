import numpy as np
import math


def create_ms_kernel(w, h):
    w = int(math.floor(w / 2))
    h = int(math.floor(h / 2))
    kernel = np.meshgrid(np.arange(-w, w + 1), np.arange(-h, h + 1))
    return kernel


def get_mean_shift_vector(w, xi, yi, kernel):
    dx = np.sum(w*xi*kernel)/np.sum(w)
    dy = np.sum(w*yi*kernel)/np.sum(w)
    return [dx, dy]
