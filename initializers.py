import numpy as np
from numpy._typing import NDArray


class Uniform:
    def __init__(self, shape: tuple):
        self.shape = shape

    def initialize(self):
        return np.random.uniform(low=-1, high=1, size=(self.shape[0], self.shape[1])) * 0.1


class Xavier:
    def __init__(self, shape: tuple):
        self.shape = shape

    def initialize(self):
        return np.random.randn(self.shape[0], self.shape[1]) * np.sqrt(6 / self.shape[0] + self.shape[1])


class He:
    def __init__(self, shape: tuple):
        self.shape = shape

    def initialize(self):
        return np.random.randn(self.shape[0], self.shape[1]) * np.sqrt(2 / self.shape[0])
