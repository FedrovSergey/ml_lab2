import os
from dotenv import load_dotenv
from .NetLay import net_lay
from .CRUD_files import *


def main(epochs, res_the_scales=False):
    img_matrix = read_img_to_matrix('./Files/alldata.csv')

    load_dotenv()
    const_width = int(os.getenv('const_width'))
    count_neurons1 = int(os.getenv('count_neurons1'))
    count_neurons2 = int(os.getenv('count_neurons2'))

    scales_index = 2

    if res_the_scales:
        create_scales_file('scales_1.csv', const_width ** 2, count_neurons1)
        create_scales_file('scales_2.csv', count_neurons1, count_neurons2)

    for _ in range(epochs):
        if img_matrix is not None:

            row = img_matrix[0].shape[0]

            for i in range(row):

                layer_matrices = img_matrix[0].iloc[i]
                for j in range(1, scales_index + 1):

                    file_path = f'scales_{j}.csv'
                    scales_matrix = read_scales_to_matrix(file_path)

                    if scales_matrix is not None:
                        layer_matrices = net_lay(layer_matrices, scales_matrix)
                    else:
                        print("Error")
                # fas


if __name__ == "__main__":
    file_path = "../Files/data_1.csv"

    matrix = read_img_to_matrix(file_path)
    print(matrix)
