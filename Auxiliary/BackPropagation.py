import numpy as np
import os
from dotenv import load_dotenv

from .Softmax import softmax
from .Sigmoid import dsigmoid_arr
from Service.CRUD_files import read_scales_to_matrix, write_scales_to_file
from .CrossEntropy import binary_cross_entropy


#from Auxiliary.Sigmoid import dsigmoid_arr
#from Auxiliary import dsigmoid_arr, binary_cross_entropy

def back_propagation(layer_matrices, true_answer):
    W3 = read_scales_to_matrix('scales_end.csv')
    dA3 = layer_matrices[3]
    answer = np.zeros(10)
    answer[true_answer] = 1
    loss = binary_cross_entropy(answer, softmax(dA3))
    dA3[true_answer] -= 1
    dW3 = np.dot(layer_matrices[2], dA3.T)

    W2 = read_scales_to_matrix('scales_2.csv')
    dA2 = np.dot(dA3.T, W3.T) * dsigmoid_arr(layer_matrices[2]).T
    dW2 = np.dot(layer_matrices[1], dA2)

    W1 = read_scales_to_matrix('scales_1.csv')
    dA1 = np.dot(dA2, W2.T) * dsigmoid_arr(layer_matrices[1]).T
    dW1 = np.dot(layer_matrices[0], dA1)

    load_dotenv()
    const_e = float(os.getenv('e'))

    W_back1 = W1 - const_e * dW1
    write_scales_to_file(W_back1, 'scales_1.csv')

    W_back2 = W2 - const_e * dW2
    write_scales_to_file(W_back2, 'scales_2.csv')

    W_back3 = W3 - const_e * dW3
    write_scales_to_file(W_back3, 'scales_end.csv')

    return loss
    # return W_back1, W_back2, W_back3
