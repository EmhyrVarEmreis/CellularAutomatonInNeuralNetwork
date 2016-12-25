import os
import sys
import time

import numpy as np

from neural import TrainingLoader
from neural.SimpleLayeredNeuralNetwork import SimpleLayeredNeuralNetwork
from neural.SimpleNeuralNetwork import SimpleNeuralNetwork

opt_tries = 1000
opt_cycles = 30001
opt_multi = True
opt_neural_num = 10
opt_every = True
opt_step = 1000
opt_max_error = 0.2734375
opt_folder = 'tmp/search/' + str(opt_neural_num)
opt_input_file = 'resource/life_all'


def main(argv):
    global opt_folder

    training_set_tmp = TrainingLoader.load(opt_input_file)

    inputs = np.array(training_set_tmp[0])
    outputs = np.array([training_set_tmp[1]]).T

    opt_folder = opt_folder + '/' + str(round(time.time()))

    os.makedirs(opt_folder, exist_ok=True)

    best_of_bests = 1

    if opt_multi:
        network = SimpleLayeredNeuralNetwork()
        network.add_layer(opt_neural_num, len(training_set_tmp[0][0]))
        network.add_layer(1, opt_neural_num)
    else:
        network = SimpleNeuralNetwork(len(training_set_tmp[0][0]))

    for i in range(opt_tries):
        print("Iteration: " + str(i + 1))

        np.random.seed()
        network.rebuild()
        last_error = 0

        best_weights = None

        for j in range(opt_cycles):
            network.train_once(inputs, outputs)
            if (j % opt_step) == 0:
                if last_error == network.error:
                    break
                if network.error < opt_max_error:
                    if network.error < last_error:
                        best_weights = network.get_weights()
                last_error = network.error

        if opt_every and best_weights:
            print("\tError: " + str(network.error))
            network.save_synaptic_weights(opt_folder + '/' + str(i) + '.txt')
        if network.error < best_of_bests:
            print("\tCurrent best error: " + str(network.error))
            best_of_bests = network.error
            network.save_synaptic_weights(opt_folder + '/' + str(network.error) + '-best.txt')


if __name__ == "__main__":
    main(sys.argv[1:])
