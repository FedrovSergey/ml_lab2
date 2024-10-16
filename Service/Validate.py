import os

from dotenv import load_dotenv

from Service import read_img_to_matrix
from Service import neuron_net


def validation(epochs):
    img_matrix = read_img_to_matrix('./Files/test.csv')

    load_dotenv()
    scales_index = int(os.getenv('scales_index'))

    for _ in range(epochs):
        if img_matrix is not None:

            row = img_matrix[0].shape[0]

            for i in range(row):

                layer_matrices = img_matrix[0].iloc[i]
                layer_matrices = neuron_net(layer_matrices, scales_index)[0]
                print(f"Result{i} for {img_matrix[1][i]}:\n{layer_matrices}")
