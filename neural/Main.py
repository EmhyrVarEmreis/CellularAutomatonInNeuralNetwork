from numpy import array

from neural import TrainingLoader
from neural.SimpleNeuralNetwork import SimpleNeuralNetwork

if __name__ == "__main__":
    training_set = TrainingLoader.load('../tmp/l1')

    neural_network = SimpleNeuralNetwork(len(training_set[0][0]))

    print("Random starting synaptic weights: ")
    print(neural_network.synaptic_weights)

    # training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    # training_set_outputs = array([[0, 1, 1, 0]]).T

    training_set_inputs = array(training_set[0])
    training_set_outputs = array([training_set[1]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("New synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    filename = '../tmp/n1'

    print("Saving as %s" % filename)
    neural_network.save_synaptic_weights(filename)

    print("Reading as %s" % filename)
    neural_network.read_synaptic_weights(filename)

    print("Weights:")
    print(neural_network.synaptic_weights)

    print("Checking learn status: ")
    pos = 0
    n = 0
    for row in training_set[0]:
        ret = neural_network.think(array(row))
        if ret > 0.5:
            ret = 1
        else:
            ret = 0
        if ret == training_set[1][n]:
            pos += 1
        n += 1
    print(str(pos / n * 100) + '%')
