import numpy as np


def binary_cross_entropy(t, p):
    t = t.reshape(t.shape[0], -1)
    return -np.sum(t * np.log(p) + (1 - t) * np.log(1 - p))