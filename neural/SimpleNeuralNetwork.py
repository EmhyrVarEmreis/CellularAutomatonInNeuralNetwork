import numpy as np


class SimpleNeuralNetwork:
    synaptic_weights = None
    input_size = None

    def __init__(self, n, filename=None):
        self.error = -1
        self.input_size = n
        if filename is not None:
            self.read_synaptic_weights(filename)
        else:
            self.rebuild()

    def rebuild(self):
        self.synaptic_weights = 2 * np.random.random((self.input_size, 1)) - 1

    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def __sigmoid_derivative(x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            self.train_once(training_set_inputs, training_set_outputs)

    def train_once(self, training_set_inputs, training_set_outputs):
        output = self.think(training_set_inputs)

        error = training_set_outputs - output

        self.error = np.mean(np.abs(error))

        # noinspection PyTypeChecker
        adjustment = np.dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

        self.synaptic_weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(np.dot(inputs, self.synaptic_weights))

    def think_output(self, inputs):
        return self.think(inputs)

    def get_weights(self):
        return self.synaptic_weights

    def read_synaptic_weights(self, filename):
        with open(filename) as f:
            self.synaptic_weights = \
                np.array([[float(i) for i in l.split()] for l in f.readlines() if
                          (not l.startswith('#')) and (not l.strip() == '')]).T

    def save_synaptic_weights(self, filename):
        with open(filename, 'w') as f:
            f.write('# ' + str(self.error) + '\n')
            f.writelines([' '.join([str(item) for item in row]) for row in self.synaptic_weights.T])

    def print_weights(self):
        print(self.synaptic_weights)

    def verify(self, training_set):
        pos = 0
        n = 0
        for row in training_set[0]:
            ret = self.think(np.array([float(i) for i in row]))
            if ret > 0.5:
                ret = 1
            else:
                ret = 0
            if ret == training_set[1][n]:
                pos += 1
            n += 1
        return pos / n * 100
