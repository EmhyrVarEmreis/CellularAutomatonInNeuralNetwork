import random

import numpy as np

from neural.SimpleLayeredNeuralNetwork import SimpleLayeredNeuralNetwork

if __name__ == "__main__":
    random.seed(1)

    neural_network = SimpleLayeredNeuralNetwork()
    neural_network.add_layer(4, 3)
    neural_network.add_layer(2, 4)
    neural_network.add_layer(1, 2)

    print("Random synaptic weights: ")
    neural_network.print_weights()

    training_set_inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
    training_set_outputs = np.array([[0, 1, 1, 1, 1, 0, 0]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("Synaptic weights after learning: ")
    neural_network.print_weights()

    print("[1, 0, 0] -> ?: ")
    print(neural_network.think_output(np.array([1, 1, 0])))

    print("Weights before saving:")
    neural_network.print_weights()

    filename = 'tmp/n1'

    print("Saving as %s" % filename)
    neural_network.save_synaptic_weights(filename)

    print("Reading as %s" % filename)
    neural_network.read_synaptic_weights(filename)

    print("Weights after reading:")
    neural_network.print_weights()

    print("[1, 0, 0] -> ?: ")
    print(neural_network.think_output(np.array([1, 1, 0])))
