from numpy import array

from neural.SimpleNeuralNetwork import SimpleNeuralNetwork

if __name__ == "__main__":
    neural_network = SimpleNeuralNetwork(3)

    print("Random synaptic weights: ")
    print(neural_network.synaptic_weights)

    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print("Synaptic weights after learning: ")
    print(neural_network.synaptic_weights)

    print("[1, 0, 0] -> ?: ")
    print(neural_network.think(array([1, 0, 0])))

    print("Weights before saving:")
    print(neural_network.synaptic_weights)

    filename = 'tmp/n1'

    print("Saving as %s" % filename)
    neural_network.save_synaptic_weights(filename)

    print("Reading as %s" % filename)
    neural_network.read_synaptic_weights(filename)

    print("Weights after reading:")
    print(neural_network.synaptic_weights)

    print("[1, 0, 0] -> ?: ")
    print(neural_network.think(array([1, 0, 0])))
