from model import NeuralNetwork


class Train:
    def __init__(self, network: NeuralNetwork, learning_rate: float, batch_size: int, epochs: int, loss_function):
        self.network = network
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss_function = loss_function

    def train(self, inputs, outputs):
        batch_count = inputs.shape[0]//self.batch_size
        cost = []
        accuracies = []
        for epoch in range(self.epochs):

            for batch_idx in range(batch_count):
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size

                input_batch = inputs[start_idx:end_idx]
                output_batch = outputs[start_idx:end_idx]

                predictions = self.network.forward(input_batch)

                loss = self.loss_function.forward(predictions, output_batch)
                accuracy = self.loss_function.accuracy(predictions, output_batch)
                cost.append(loss.mean())
                accuracies.append(accuracy)

                self.loss_function.backward(predictions, output_batch)
                self.network.backward(self.loss_function.d_input)
                self.optimise()

                if epoch % 500 == 0:
                    print(f"{epoch}. Accuracy: {accuracy} Loss: {loss.mean()}")

        return cost, accuracies


    def optimise(self):
        for layer in self.network.layers:
            layer.weights -= self.learning_rate*layer.d_weights
            layer.bias = layer.bias.reshape(1, -1)
            layer.bias -= self.learning_rate*layer.d_bias


