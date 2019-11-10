import numpy as np
import random

class HopfieldNetwork:

    def __init__(self, learning_rate, max_iter):
        self.Weights = None

        self.activation = lambda x: np.sign(x)
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def train(self, X):
        self.Weights = self.__hebbianRule(X, X.shape[1], X.shape[0])

    def __hebbianRule(self, X, N, M):
        dot = np.dot(X.T, X)
        return 1 / N * (dot - M * np.identity(N))

    def reconstruct_sync(self, x):
        previous_steps = []
        for i in range(0, self.max_iter):
            new_x = np.dot(self.Weights, x)
            new_x = self.activation(new_x)
            if (new_x == x).all():
                return new_x
            # if previous_steps.__contains__(new_x):
            #     print("Cycle detected")
            #     return []
            previous_steps.append(x)
            x = new_x

    def reconstruct_async(self, x):
        previous_steps = []
        new_x = x
        for i in range(0, self.max_iter * len(x)):
            n = random.randint(0, len(x))

            val = np.dot(new_x, self.Weights[n])

            new_x[n] = self.activation(val)

            if (new_x == x).all():
                return new_x

            previous_steps.append(new_x)
            x = new_x
