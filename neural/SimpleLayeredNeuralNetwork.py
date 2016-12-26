import numpy as np


class NeuronLayer:
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.number_of_neurons = number_of_neurons
        self.number_of_inputs_per_neuron = number_of_inputs_per_neuron
        self.synaptic_weights = 2 * np.random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


class SimpleLayeredNeuralNetwork:
    def __init__(self):
        self.layers = []
        self.error = -1

    def add_layer(self, number_of_neurons, number_of_inputs_per_neuron):
        self.layers.append(NeuronLayer(number_of_neurons, number_of_inputs_per_neuron))

    def rebuild(self):
        new_layers = []
        for layer in self.layers:
            new_layers.append(NeuronLayer(layer.number_of_neurons, layer.number_of_inputs_per_neuron))
        self.layers = new_layers

    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def __sigmoid_derivative(x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            self.train_once(training_set_inputs, training_set_outputs)

    # noinspection PyUnresolvedReferences
    def train_once(self, training_set_inputs, training_set_outputs):
        output_from_layer_1, output_from_layer_2 = self.think(training_set_inputs)

        layer2_error = training_set_outputs - output_from_layer_2
        layer2_delta = layer2_error * self.__sigmoid_derivative(output_from_layer_2)

        self.error = np.mean(np.abs(layer2_error))

        layer1_error = layer2_delta.dot(self.layers[1].synaptic_weights.T)
        layer1_delta = layer1_error * self.__sigmoid_derivative(output_from_layer_1)

        self.layers[0].synaptic_weights += training_set_inputs.T.dot(layer1_delta)
        self.layers[1].synaptic_weights += output_from_layer_1.T.dot(layer2_delta)

    # noinspection PyTypeChecker
    def think_layer(self, inputs, layer):
        return self.__sigmoid(np.dot(inputs, layer.synaptic_weights))

    # noinspection PyTypeChecker, PyUnresolvedReferences
    def think_output(self, inputs):
        return self.think(inputs)[1][0]

    # noinspection PyTypeChecker
    def think(self, inputs):
        output_from_layer1 = self.think_layer(inputs, self.layers[0])
        output_from_layer2 = self.think_layer(output_from_layer1, self.layers[1])
        return output_from_layer1, output_from_layer2

    def get_weights(self):
        return [layer.synaptic_weights for layer in self.layers]

    def print_weights(self):
        n = 0
        for layer in self.layers:
            n += 1
            print('Layer ' + str(n) + ' (' + str(layer.number_of_neurons) + ' neurons, each with ' + str(
                layer.number_of_inputs_per_neuron) + ' inputs): ')
            print(layer.synaptic_weights)

    def read_synaptic_weights(self, filename):
        with open(filename) as f:
            self.layers = []
            for l in f.readlines():
                if not l.startswith('#') and not l.strip() == '':
                    if l.startswith('&'):
                        size = [int(i) for i in l[1:].strip().split()]
                        self.layers.append(NeuronLayer(*size))
                        self.layers[-1].synaptic_weights = []
                    else:
                        if len(self.layers) > 0:
                            self.layers[-1].synaptic_weights.append([float(i) for i in l.split()])
            for layer in self.layers:
                layer.synaptic_weights = np.array(layer.synaptic_weights).T

    def save_synaptic_weights(self, filename):
        with open(filename, 'w') as f:
            f.write('# ' + str(self.error) + '\n')
            for layer in self.layers:
                f.write('& ' + str(layer.number_of_neurons) + ' ' + str(layer.number_of_inputs_per_neuron) + '\n')
                f.writelines([(' '.join([str(item) for item in row])) + '\n' for row in layer.synaptic_weights.T])

    def verify(self, training_set):
        pos = 0
        n = 0
        for row in training_set[0]:
            ret = self.think_output(np.array([float(i) for i in row]))
            if ret > 0.5:
                ret = 1
            else:
                ret = 0
            if ret == training_set[1][n]:
                pos += 1
            n += 1
        return pos / n * 100
