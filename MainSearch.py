import os
import sys

import numpy as np

from neural import TrainingLoader
from neural.SimpleLayeredNeuralNetwork import SimpleLayeredNeuralNetwork


def main(argv):
    opt_tries = 1000
    opt_cycles = 30001
    opt_multi_neural = '9-2'
    opt_every = True
    opt_step = 1000
    opt_max_error = 0.2734375
    opt_mutations = 5
    opt_mutation_value = .05
    opt_folder = 'tmp/search'
    opt_input_file = 'resource/life_all'
    opt_weights_file = None

    # TODO Add specific settings to enable learning from weights file with mutations

    if len(argv) > 0:
        opt_tries = int(argv[0])
        opt_cycles = int(argv[1])
        opt_multi_neural = argv[2]
        opt_every = str(argv[3]).lower() == 't'
        opt_step = int(argv[4])
        opt_max_error = float(argv[5])
        opt_mutations = int(argv[6]) if argv[6].isdigit() else 0
        opt_mutation_value = float(argv[7])
        if len(argv) > 8:
            opt_weights_file = argv[8]

    training_set_tmp = TrainingLoader.load(opt_input_file)

    inputs = np.array(training_set_tmp[0])
    outputs = np.array([training_set_tmp[1]]).T

    best_of_bests = 1

    network = SimpleLayeredNeuralNetwork()

    opt_multi_neural = opt_multi_neural.split('-')

    if len(opt_multi_neural) == 1 and int(opt_multi_neural[0]) == 1:
        network.add_layer(1, len(training_set_tmp[0][0]))
    else:
        network.add_layer(int(opt_multi_neural[0]), len(training_set_tmp[0][0]))
        for idx, val in enumerate(opt_multi_neural[1:]):
            network.add_layer(int(val), int(opt_multi_neural[idx]))
        network.add_layer(1, int(opt_multi_neural[-1]))

    opt_folder = opt_folder + '/' + '.'.join(
        [str(layer.number_of_neurons) + '-' + str(layer.number_of_inputs_per_neuron) for layer in network.layers]
    )

    os.makedirs(opt_folder, exist_ok=True)

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
                    if opt_mutations == 0:
                        print('\tBreak - no progress')
                        break
                    else:
                        print('\tMutating %d weights' % opt_mutations)
                        for x in range(opt_mutations):
                            layer = np.random.choice(network.layers)
                            num = np.random.randint(0, len(layer.synaptic_weights))
                            nun = np.random.randint(0, len(layer.synaptic_weights[num]))
                            if np.random.choice([True, False]):
                                layer.synaptic_weights[num][nun] *= (1.0 + opt_mutation_value)
                            else:
                                layer.synaptic_weights[num][nun] *= (1.0 - opt_mutation_value)
                if network.error < opt_max_error:
                    if network.error < last_error:
                        best_weights = network.get_weights()
                last_error = network.error

        print('\tCurrent error: ' + str(network.error))

        if opt_every and best_weights:
            print("\tError: " + str(network.error))
            network.save_synaptic_weights(opt_folder + '/' + str(network.error) + '.txt')
        if network.error < best_of_bests and network.error < opt_max_error:
            print("\tCurrent best error: " + str(network.error))
            best_of_bests = network.error
            network.save_synaptic_weights(opt_folder + '/' + str(network.error) + '.txt')


if __name__ == "__main__":
    main(sys.argv[1:])
