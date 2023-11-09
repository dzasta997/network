import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from activations import ReLU, Softmax, Sigmoid, TanH
from dataset import heart_classification
from evaluation import BCE
from initializers import He, Xavier
from model import NeuralNetwork
from train import Train

if __name__ == "__main__":
    inputs, outputs = heart_classification()
    inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs, outputs, test_size=0.16)

    learning_rate = 0.02
    batch_size = inputs_train.shape[0]
    epochs = 5000
    loss_function = BCE()

    if len(outputs_train.shape) == 1:
        outputs_train = np.reshape(outputs_train, (outputs_train.shape[0], 1))
    input_dim = inputs_train.shape[1]
    hidden_dim = inputs_train.shape[0]
    output_dim = outputs_train.shape[1]
    num_of_hidden_layers = 1

    activation = ReLU()
    activation_output = Sigmoid()

    initialization = He
    output_initialization = Xavier
    network = NeuralNetwork(input_dim,
                            hidden_dim,
                            output_dim,
                            num_of_hidden_layers,
                            activation,
                            activation_output,
                            initialization,
                            output_initialization)

    train = Train(network, learning_rate, batch_size, epochs, loss_function)
    cost, accuracies = train.train(inputs_train, outputs_train)
    plt.plot([a for a in range(len(cost))],cost)
    plt.show()


