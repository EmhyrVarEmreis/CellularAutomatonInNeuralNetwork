import sys

from neural.SimpleLayeredNeuralNetwork import SimpleLayeredNeuralNetwork
from util.training import load_training_set


def main(argv):
    opt_neural_network_file = 'resource/learned/0.0108333399636.txt'
    opt_training_set_location = 'resource/training_set/life_all'
    if len(argv) == 2:
        opt_neural_network_file = argv[0]
        opt_training_set_location = argv[1]

    nn = SimpleLayeredNeuralNetwork()
    nn.read_synaptic_weights(opt_neural_network_file)
    print(str(nn.verify(load_training_set(opt_training_set_location))) + '%')


if __name__ == "__main__":
    main(sys.argv[1:])
