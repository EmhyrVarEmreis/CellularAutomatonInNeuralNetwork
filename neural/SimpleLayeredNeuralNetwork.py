import numpy as np


class NeuronLayer:
    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):
        self.number_of_neurons = number_of_neurons
        self.number_of_inputs_per_neuron = number_of_inputs_per_neuron
        self.synaptic_weights = 2 * np.random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1


# TODO Make true multi-layered
class SimpleLayeredNeuralNetwork:
    def __init__(self):
        self.layers = []
        self.error = -1
        self.train_factor = 1.0

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

    # noinspection PyUnresolvedReferences,PyTypeChecker
    def train_once(self, training_set_inputs, training_set_outputs):
        self.think(training_set_inputs)

        prev_layer = None
        for i, layer in reversed(list(enumerate(self.layers))):
            next_layer = None if i == 0 else self.layers[i - 1]
            if prev_layer is None:
                layer.error = training_set_outputs - layer.output
                self.error = np.mean(np.abs(layer.error))
            else:
                layer.error = np.dot(prev_layer.delta, prev_layer.synaptic_weights.T)
            layer.delta = layer.error * self.__sigmoid_derivative(layer.output)
            if next_layer is None:
                layer.adjustment = self.train_factor * np.dot(training_set_inputs.T, layer.delta)
            else:
                layer.adjustment = self.train_factor * np.dot(next_layer.output.T, layer.delta)
            prev_layer = layer

        for layer in self.layers:
            layer.synaptic_weights += layer.adjustment

    # noinspection PyTypeChecker
    def think_layer(self, inputs, layer):
        layer.output = self.__sigmoid(np.dot(inputs, layer.synaptic_weights))
        return layer.output

    # noinspection PyTypeChecker, PyUnresolvedReferences
    def think_output(self, inputs):
        return self.think(inputs)[-1][0]

    # noinspection PyTypeChecker
    def think(self, inputs):
        outputs = [self.think_layer(inputs, self.layers[0])]
        for layer in self.layers[1:]:
            outputs.append(self.think_layer(outputs[-1], layer))
        return outputs

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
