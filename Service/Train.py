import os

import numpy as np
from dotenv import load_dotenv

from Auxiliary import back_propagation, binary_cross_entropy, softmax
from .NeuronNet import neuron_net
from .CRUD_files import *


def train(epochs, res_the_scales=False):
    img_matrix = read_img_to_matrix('./Files/training.csv')

    load_dotenv()
    const_width = int(os.getenv('const_width'))
    count_neurons1 = int(os.getenv('count_neurons1'))
    count_neurons2 = int(os.getenv('count_neurons2'))
    count_class = int(os.getenv('count_class'))
    scales_index = int(os.getenv('scales_index'))

    if res_the_scales:
        create_scales_file('scales_1.csv', const_width ** 2, count_neurons1)
        create_scales_file('scales_2.csv', count_neurons1, count_neurons2)
        create_scales_file('scales_end.csv', count_neurons2, count_class)

    for epoch in range(epochs):
        if img_matrix is not None:

            number_err = np.empty(0)
            count_true = 0
            data_for_metrics = np.zeros((count_class, 4))
            precision_arr = np.zeros(count_class)
            recall_arr = np.zeros(count_class)
            accuracy_arr = np.zeros(count_class)

            row = img_matrix[0].shape[0]

            for i in range(row):

                layer_matrices = img_matrix[0].iloc[i]
                end_y, layer_matrices = neuron_net(layer_matrices, scales_index)
                true_answer = img_matrix[1][i][0] - 1

                get_answer = end_y.idxmax()
                for j in range(count_class):
                    if get_answer == j and true_answer == j:
                        data_for_metrics[j][0] += 1
                    elif get_answer != j and true_answer == j:
                        data_for_metrics[j][1] += 1
                    elif get_answer == j and true_answer != j:
                        data_for_metrics[j][2] += 1
                    elif get_answer != j and true_answer != j:
                        data_for_metrics[j][3] += 1
               # if get_answer == true_answer:
                #    count_true += 1

                number_err.append(back_propagation(layer_matrices, true_answer))

            for i in range(count_class):
                precision_arr[i] += data_for_metrics[i][0] / (data_for_metrics[i][0] + data_for_metrics[i][2])
                recall_arr[i] += data_for_metrics[i][0] / (data_for_metrics[i][0] + data_for_metrics[i][1])
                accuracy_arr[i] += (data_for_metrics[i][0] + data_for_metrics[i][3]) / (
                        data_for_metrics[i][0] + data_for_metrics[i][1] + data_for_metrics[i][2] +
                        data_for_metrics[i][3])

            loss = np.mean(number_err)
            accuracy = np.mean(accuracy_arr)
            precision = np.mean(precision_arr)
            recall = np.mean(recall_arr)

            metric_arr = np.array([loss, accuracy, precision, recall])

            write_arr_to_file(metric_arr, './Files/metrics.csv')
        print(f"Later {epoch+1} epochs.")


def train_validate_for_metrics(epochs, res_the_scales=False):
    img_matrix = read_img_to_matrix('./Files/training.csv')
    img_matrix1 = read_img_to_matrix('./Files/test.csv')

    load_dotenv()
    const_width = int(os.getenv('const_width'))
    count_neurons1 = int(os.getenv('count_neurons1'))
    count_neurons2 = int(os.getenv('count_neurons2'))
    count_class = int(os.getenv('count_class'))
    scales_index = int(os.getenv('scales_index'))

    if res_the_scales:
        create_scales_file('scales_1.csv', const_width ** 2, count_neurons1)
        create_scales_file('scales_2.csv', count_neurons1, count_neurons2)
        create_scales_file('scales_end.csv', count_neurons2, count_class)

    for epoch in range(epochs):
        if img_matrix is not None:

            number_err = []
            #count_true = 0
            data_for_metrics = np.zeros((count_class, 4))
            precision_arr = np.zeros(count_class)
            recall_arr = np.zeros(count_class)
            accuracy_arr = np.zeros(count_class)

            row = img_matrix[0].shape[0]

            for i in range(row):

                layer_matrices = img_matrix[0].iloc[i]
                end_y, layer_matrices = neuron_net(layer_matrices, scales_index)
                true_answer = img_matrix[1][i][0] - 1

                get_answer = end_y.idxmax()
                for j in range(count_class):
                    if get_answer == j and true_answer == j:
                        data_for_metrics[j][0] += 1
                    elif get_answer != j and true_answer == j:
                        data_for_metrics[j][1] += 1
                    elif get_answer == j and true_answer != j:
                        data_for_metrics[j][2] += 1
                    elif get_answer != j and true_answer != j:
                        data_for_metrics[j][3] += 1
                #if get_answer == true_answer:
                 #   count_true += 1

                number_err.append(back_propagation(layer_matrices, true_answer))

            for i in range(count_class):
                if (data_for_metrics[i][0] + data_for_metrics[i][2]) != 0:
                    precision_arr[i] += data_for_metrics[i][0] / (data_for_metrics[i][0] + data_for_metrics[i][2])
                if (data_for_metrics[i][0] + data_for_metrics[i][1]) != 0:
                    recall_arr[i] += data_for_metrics[i][0] / (data_for_metrics[i][0] + data_for_metrics[i][1])
                if ((data_for_metrics[i][0] + data_for_metrics[i][1] + data_for_metrics[i][2] + data_for_metrics[i][3])) != 0:
                    accuracy_arr[i] += (data_for_metrics[i][0] + data_for_metrics[i][3]) / (
                            data_for_metrics[i][0] + data_for_metrics[i][1] + data_for_metrics[i][2] +
                            data_for_metrics[i][3])

            loss = np.mean(number_err)
            accuracy = np.mean(accuracy_arr)
            precision = np.mean(precision_arr)
            recall = np.mean(recall_arr)

            metric_arr = np.array([loss, accuracy, precision, recall])

            write_arr_to_file(metric_arr, './Files/metrics.csv')


        if img_matrix1 is not None:

            number_err = []
            #count_true = 0
            data_for_metrics = np.zeros((count_class, 4))
            precision_arr = np.zeros(count_class)
            recall_arr = np.zeros(count_class)
            accuracy_arr = np.zeros(count_class)

            row = img_matrix1[0].shape[0]

            for i in range(row):

                layer_matrices = img_matrix1[0].iloc[i]
                end_y, layer_matrices = neuron_net(layer_matrices, scales_index)
                #print(f"Result{i} for {img_matrix1[1][i]}:\n{layer_matrices}")
                true_answer = img_matrix1[1][i][0] - 1

                get_answer = end_y.idxmax()
                for j in range(count_class):
                    if get_answer == j and true_answer == j:
                        data_for_metrics[j][0] += 1
                    elif get_answer != j and true_answer == j:
                        data_for_metrics[j][1] += 1
                    elif get_answer == j and true_answer != j:
                        data_for_metrics[j][2] += 1
                    elif get_answer != j and true_answer != j:
                        data_for_metrics[j][3] += 1
                #if get_answer == true_answer:
                 #   count_true += 1

                answer = np.zeros(10)
                answer[true_answer] = 1
                loss = binary_cross_entropy(answer, softmax(layer_matrices[3]))
                number_err.append(loss)

            for i in range(count_class):
                if (data_for_metrics[i][0] + data_for_metrics[i][2]) != 0:
                    precision_arr[i] += data_for_metrics[i][0] / (data_for_metrics[i][0] + data_for_metrics[i][2])
                if (data_for_metrics[i][0] + data_for_metrics[i][1]) != 0:
                    recall_arr[i] += data_for_metrics[i][0] / (data_for_metrics[i][0] + data_for_metrics[i][1])
                if ((data_for_metrics[i][0] + data_for_metrics[i][1] + data_for_metrics[i][2] + data_for_metrics[i][3])) != 0:
                    accuracy_arr[i] += (data_for_metrics[i][0] + data_for_metrics[i][3]) / (
                            data_for_metrics[i][0] + data_for_metrics[i][1] + data_for_metrics[i][2] +
                            data_for_metrics[i][3])

            loss = np.mean(number_err)
            accuracy = np.mean(accuracy_arr)
            precision = np.mean(precision_arr)
            recall = np.mean(recall_arr)

            metric_arr = np.array([loss, accuracy, precision, recall])

            write_arr_to_file(metric_arr, './Files/metrics_valid.csv')

        print(f"Later {epoch + 1} epochs.")

if __name__ == "__main__":
    file_path = "../Files/data_1.csv"

    matrix = read_img_to_matrix(file_path)
    print(matrix)
