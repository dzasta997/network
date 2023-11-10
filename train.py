from model import NeuralNetwork
import statistics

class Train:
    def __init__(self, network: NeuralNetwork, learning_rate: float, batch_size: int, epochs: int, loss_function):
        self.network = network
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_function = loss_function

    def train(self, inputs, outputs):
        batch_count = inputs.shape[0] // self.batch_size
        cost = []
        accuracies = []
        for epoch in range(self.epochs):
            batch_loss = []
            batch_accuracy = []
            for batch_idx in range(batch_count):
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size

                input_batch = inputs[start_idx:end_idx]
                output_batch = outputs[start_idx:end_idx]

                predictions = self.network.forward(input_batch)

                loss = self.loss_function.forward(predictions, output_batch)
                accuracy = self.loss_function.accuracy(predictions, output_batch)

                batch_loss.append(loss)
                batch_accuracy.append(accuracy)

                self.loss_function.backward(predictions, output_batch)
                self.network.backward(self.loss_function.d_input)
                self.optimise(input_batch.shape[0])

                if epoch % 500 == 0:
                    print(f"{epoch}. Accuracy: {accuracy} Loss: {loss.mean()}")

            cost.append(statistics.mean(batch_loss))
            accuracies.append(statistics.mean(batch_accuracy))




        return cost, accuracies

    def optimise(self, batch_size):
        for layer in self.network.layers:
            layer.weights -= self.learning_rate * layer.d_weights
            layer.bias = self.learning_rate * layer.d_bias.T
