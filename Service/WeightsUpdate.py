import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

from .CRUD_files import read_scales_to_matrix
from .Sigmoid import sigmoid

def backPropagation(layer_matrices, true_answer, x):
    W3 = read_scales_to_matrix('scales_end.csv')
    dA3 = layer_matrices[3] - true_answer
    dW3 = np.dot(layer_matrices[2].T, dA3)

    W2 = read_scales_to_matrix('scales_2.csv')
    dA2 = np.dot(dA3, W3.T) * sigmoid(layer_matrices[2])
    dW2 = np.dot(layer_matrices[1].T, dA2)

    W1 = read_scales_to_matrix('scales_1.csv')
    dA1 = np.dot(dA2, W2.T) * sigmoid(layer_matrices[1])
    dW1 = np.dot(x.T, dA1)

    load_dotenv()
    const_e = int(os.getenv('e'))
    W_back1 = W1 - const_e * dW1
    W_back2 = W2 - const_e * dW2
    W_back3 = W3 - const_e * dW3

    return W_back1, W_back2, W_back3
