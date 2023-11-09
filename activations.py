import numpy as np


class ReLU:  # This one is ok
    def __init__(self):
        self.input = None
        self.output = None
        self.d_input = None

    def forward(self, inputs):
        self.input = inputs
        self.output = np.maximum(0, inputs)
        return self.output

    def backward(self, d_input):
        self.d_input = d_input.copy()
        self.d_input[self.input <= 0] = 0
        return self.d_input


class TanH:
    def __init__(self):
        self.input = None
        self.output = None
        self.d_input = None

    def forward(self, inputs):
        self.input = inputs
        self.output = np.tanh(inputs)
        return self.output

    def backward(self, d_inputs):
        self.d_input = 1 - np.square(np.tanh(d_inputs))
        return self.d_input


class Sigmoid:
    def __init__(self):
        self.input = None
        self.output = None
        self.d_input = None

    def forward(self, inputs):
        self.input = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, d_inputs):
        sigmoid = 1 / (1 + np.exp(-self.input))
        self.d_input = d_inputs*sigmoid*(1-sigmoid)
        return self.d_input

class Softmax:
    def __init__(self):
        self.input = None
        self.output = None
        self.d_input = None

    def forward(self, inputs):
        self.input = inputs
        exp = np.exp(inputs)
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def backward(self, d_values):
        d_inputs = np.empty_like(d_values)

        for idx, (output, d_value) in enumerate(zip(self.output, d_values)):
            # Flatten output
            output = output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(output) - np.dot(output, output.T)

            d_inputs[idx] = np.dot(jacobian_matrix, d_value)

        self.d_input = d_inputs
        return self.d_input
