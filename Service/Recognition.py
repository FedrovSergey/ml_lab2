import os

from dotenv import load_dotenv

from .NeuronNet import neuron_net


def recognition(img):
    layer_matrices = img

    load_dotenv()
    scales_index = int(os.getenv('scales_index'))

    recognizer = neuron_net(layer_matrices, scales_index)[0]
    return max(recognizer)
