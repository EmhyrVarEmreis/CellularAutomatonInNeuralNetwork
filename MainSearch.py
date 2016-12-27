import os
import sys

import numpy as np

from neural import TrainingLoader
from neural.SimpleLayeredNeuralNetwork import SimpleLayeredNeuralNetwork
from neural.SimpleNeuralNetwork import SimpleNeuralNetwork


def main(argv):
    opt_tries = 1000
    opt_cycles = 30001
    opt_multi = True
    opt_multi_neural_num = 10
    opt_every = True
    opt_step = 1000
    opt_max_error = 0.2734375
    opt_folder = 'tmp/search/'
    opt_input_file = 'resource/life_all'
    opt_weights_file = None

    # TODO Add specific settings to enable learning from weights file with mutations

    if len(argv) > 0:
        opt_tries = int(argv[0])
        opt_cycles = int(argv[1])
        opt_multi = str(argv[2]).lower() == 't'
        opt_multi_neural_num = int(argv[3])
        opt_every = str(argv[4]).lower() == 't'
        opt_step = int(argv[5])
        opt_max_error = float(argv[6])
        if len(argv) > 7:
            opt_weights_file = argv[7]

    training_set_tmp = TrainingLoader.load(opt_input_file)

    inputs = np.array(training_set_tmp[0])
    outputs = np.array([training_set_tmp[1]]).T

    os.makedirs(opt_folder, exist_ok=True)

    best_of_bests = 1

    if opt_multi:
        network = SimpleLayeredNeuralNetwork()
        network.add_layer(opt_multi_neural_num, len(training_set_tmp[0][0]))
        network.add_layer(1, opt_multi_neural_num)
    else:
        network = SimpleNeuralNetwork(len(training_set_tmp[0][0]))

    opt_folder = opt_folder + '/' + '.'.join(
        [str(layer.number_of_neurons) + '-' + str(layer.number_of_inputs_per_neuron) for layer in network.layers]
    )

    last_network = None

    for i in range(opt_tries):
        print("Iteration: " + str(i + 1))

        np.random.seed()
        if opt_weights_file is None:
            network.rebuild()
        else:
            network.read_synaptic_weights(opt_weights_file)

        last_error = 0

        best_weights = None

        for j in range(opt_cycles):
            network.train_once(inputs, outputs)
            if (j % opt_step) == 0:
                if last_error == network.error:
                    # TODO Make mutation instead of breaking
                    print('\tBreak - no progress')
                    break
                if network.error < opt_max_error:
                    if network.error < last_error:
                        best_weights = network.get_weights()
                last_error = network.error

        last_network = network

        if opt_every and best_weights:
            print("\tError: " + str(network.error))
            network.save_synaptic_weights(opt_folder + '/' + str(network.error) + '.txt')
        if network.error < best_of_bests and network.error < opt_max_error:
            print("\tCurrent best error: " + str(network.error))
            best_of_bests = network.error
            network.save_synaptic_weights(opt_folder + '/' + str(network.error) + '.txt')


if __name__ == "__main__":
    main(sys.argv[1:])
