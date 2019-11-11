import numpy as np
import random

class HopfieldNetwork:

    def __init__(self, learning_rate, max_iter):
        self.Weights = None

        self.activation = lambda x: np.sign(x)
        self.learning_rate = learning_rate
        self.max_iter = max_iter

        self.previous_steps = []

    def train(self, X):
        self.Weights = self.__hebbianRule(X, X.shape[1], X.shape[0])

    def __hebbianRule(self, X, N, M):
        dot = np.dot(X.T, X)
        return 1 / N * (dot - M * np.identity(N))

    def reconstruct_sync(self, x):
        x = x.copy()
        self.previous_steps = [x]
        for i in range(0, self.max_iter):
            new_x = np.dot(self.Weights, x)
            new_x = self.activation(new_x)

            self.previous_steps.append(new_x)
            if np.array_equal(new_x, x):
                return new_x

            if any(np.array_equal(v, new_x) for v in self.previous_steps):
                print("Cycle detected!")
                return new_x

            x = new_x.copy()

        print("Network did not converge!")
        return x

    def reconstruct_async(self, x):

        self.previous_steps = [x]
        x = x.copy()

        for i in range(0, self.max_iter):

            n = int(round(random.uniform(0, len(x) - 1)))
            val = np.dot(x, self.Weights[n])

            x[n] = self.activation(val)

            self.previous_steps.append(x)

            if len(self.previous_steps) > 10 * len(x) and all(np.array_equal(p, x)
                                                              for p in self.previous_steps[-10 * len(x) :]):
                return x

        print("Network did not converge!")
        return x
