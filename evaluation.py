import numpy as np


class MSE:
    def __init__(self):
        self.d_input = None

    def forward(self, actual, expected):
        return 0.5 * np.sum((actual - expected) ** 2)

    def backward(self, actual, expected):
        self.d_input = actual - expected
        return self.d_input

    def accuracy(self, actual, expected):
        return None


class BCE:
    def __init__(self):
        self.d_input = None

    def forward(self, actual, expected, eps=10e-10):
        actual = np.clip(actual, eps, 1 - eps)
        loss = -(expected * np.log(actual) + (1 - expected) * np.log(1 - actual)).mean()
        return loss

    def backward(self, actual, expected, eps=10e-10):
        actual = np.clip(actual, eps, 1 - eps)
        # self.d_input = np.divide((actual - expected), (np.multiply(actual, (1 - actual))))
        self.d_input = np.divide(1-expected, 1-actual) - np.divide(expected, actual)
        return self.d_input

    def accuracy(self, actual, expected):
        actual_category = self.convert_prob_into_class(actual)
        return np.mean(actual_category == expected)

    def convert_prob_into_class(self, probabilities):
        probs_ = np.copy(probabilities)
        probs_[probs_ > 0.5] = 1
        probs_[probs_ <= 0.5] = 0
        return probs_
