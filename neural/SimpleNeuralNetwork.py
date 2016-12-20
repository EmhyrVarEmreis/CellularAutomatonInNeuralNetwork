from numpy import exp, array, random, dot


class SimpleNeuralNetwork:
    synaptic_weights = None
    input_size = None

    def __init__(self, n, filename=None):
        self.input_size = n
        if filename is not None:
            self.read_synaptic_weights(filename)
        else:
            random.seed(1)
            self.synaptic_weights = 2 * random.random((self.input_size, 1)) - 1

    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def __sigmoid_derivative(x):
        return x * (1 - x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            output = self.think(training_set_inputs)

            error = training_set_outputs - output

            # noinspection PyTypeChecker
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            self.synaptic_weights += adjustment

    def think(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

    def read_synaptic_weights(self, filename):
        with open(filename) as f:
            self.synaptic_weights = \
                array([[float(i) for i in l.split()] for l in f.readlines() if
                       (not l.startswith('#')) and (not l.strip() == '')]).T

    def save_synaptic_weights(self, filename):
        with open(filename, 'w') as f:
            f.writelines([' '.join([str(item) for item in row]) for row in self.synaptic_weights.T])

    def verify(self, training_set):
        pos = 0
        n = 0
        for row in training_set[0]:
            ret = self.think(array([float(i) for i in row]))
            if ret > 0.5:
                ret = 1
            else:
                ret = 0
            if ret == training_set[1][n]:
                pos += 1
            n += 1
        return pos / n * 100
