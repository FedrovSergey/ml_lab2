import numpy as np


def binary_cross_entropy(t, p):
    return -np.sum(t * np.log(p) + (1 - t) * np.log(1 - p))