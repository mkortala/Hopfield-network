from enum import Enum

import numpy as np
import random


class LearningType(Enum):
    Hebbian = 1
    Ojas = 2


class HopfieldNetwork:

    def __init__(self, learning_rate, max_iter, learningType):
        self.Weights = None

        self.activation = lambda x: np.sign(x)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.learningType = learningType

    def train(self, X):
        if self.learningType == LearningType.Hebbian:
            self.Weights = self.__hebbianRule(X, X.shape[1], X.shape[0])
        else:
            self.Weights = self.__ojaRule(X, X.shape[1], X.shape[0])

    def __hebbianRule(self, X, N, M):
        dot = np.dot(X.T, X)
        return 1 / N * (dot - M * np.identity(N))

    def __ojaRule(self, X, N, M):
        self.Weights = np.outer(X[0], X[0]) / 1.
        for i in range(1, len(X)):
            dot = np.dot(self.Weights,  X[i])
            V = dot
            V = self.activation(V)
            dot2 = self.activation(np.dot(self.Weights, V))
            diff = (X[i] - dot2)
            self.Weights += self.learning_rate * np.dot(V, diff)
        return self.Weights

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
