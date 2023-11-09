import numpy as np
from numpy._typing import NDArray


# Add some initializers
class Activator:
    def __init__(self):
        self.input = None
        self.output = None


class Layer:
    def __init__(self,
                 input_size: int,
                 neurons: int,
                 initializer):
        self.neurons = neurons
        self.input_size = input_size
        self.initializer = initializer((self.input_size, self.neurons))

        self.weights = self.initializer.initialize()
        self.bias = np.zeros(neurons)
        self.input = None
        self.output = None
        self.d_weights = None
        self.d_bias = None
        self.d_input = None

        # take input layer and return its linear function dependent on weights and bias

    def forward(self, input_matrix: NDArray):
        self.input = input_matrix
        self.output = np.dot(self.input, self.weights)
        return self.output

    def backward(self, d_error):
        self.d_weights = np.dot(self.input.T, d_error)
        self.d_bias = np.sum(d_error, axis=0, keepdims=True)
        self.d_input = np.dot(d_error, self.weights.T)  # weights.t but the dim does nt go
        return self.d_input


class NeuralNetwork:
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 number_of_hidden_layers: int,
                 activation,  # TODO: Add defaults
                 output_activation,
                 initialization,
                 output_initialization):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_of_hidden_layers = number_of_hidden_layers
        self.layers = self.initialize(initialization, output_initialization)
        self.activations = self.init_activations(activation, output_activation)
        self.layers = self.initialize(initialization, output_initialization)

    def initialize(self, initialization, output_initialization):
        layers_list = [Layer(self.input_dim, self.hidden_dim, initialization)]
        for _ in range(self.num_of_hidden_layers):
            layers_list.append(Layer(self.hidden_dim, self.hidden_dim, initialization))
        layers_list.append(Layer(self.hidden_dim, self.output_dim, output_initialization))
        return layers_list

    def init_activations(self, activation, output_activation):
        act_list = []
        for _ in range(self.num_of_hidden_layers + 1):
            act_list.append(activation)
        act_list.append(output_activation)
        return act_list

    def forward(self, input_matrix):
        matrix = input_matrix
        for layer, activation in zip(self.layers, self.activations):
            layer.forward(matrix)
            activation.forward(layer.output)
            matrix = activation.output

        return matrix

    def backward(self, d_loss):
        grad = d_loss
        # print(f"Gradient: {grad.mean()}")
        for layer, activation in zip(reversed(self.layers), reversed(self.activations)):
            activation.backward(grad)
            layer.backward(activation.d_input)
            grad = layer.d_input
