from enum import Enum

import numpy as np
import random


class LearningType(Enum):
    Hebbian = 1
    Ojas = 2


class HopfieldNetwork:

    def __init__(self, learning_rate, max_iter, learning_type, epsilon):
        self.Weights = None
        self.activation = lambda x: np.sign(x)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.learningType = learning_type
        self.epsilon = epsilon

        self.previous_steps = []

    def train(self, X):
        if self.learningType == LearningType.Hebbian:
            self.Weights = self.__hebbianRule(X, X.shape[1], X.shape[0])
        else:
            self.Weights = self.__ojaRule3(X, X.shape[1], X.shape[0])

    def __hebbianRule(self, X, N, M):
        dot = np.dot(X.T, X)
        return 1 / N * (dot - M * np.identity(N))

    def __ojaRule(self, X, N, M):
        np.random.seed(30)
        self.Weights = np.zeros((N,N))

        Weights_prev = np.ones((N,N))
        while np.linalg.norm(self.Weights - Weights_prev) > self.epsilon:
            Weights_prev = self.Weights.copy()
            for i in range(0, len(X)):
                y = X[i] if i ==0 else np.dot(self.Weights, np.transpose([X[i]]))
                y = self.activation(y)
                self.Weights += self.learning_rate * y * (X[i] - y*self.Weights)
                self.__make_diagonal_zero()
        return self.Weights


    def __ojaRule2(self, X, N, M):
        weights = np.zeros((N, N))
        prev_weights = np.ones((N, N)) / N

        iter_count = self.max_iter * 100
        iter_num = 0
        while iter_num < iter_count and np.linalg.norm(weights - prev_weights) > self.epsilon:
            prev_weights = weights.copy()
            iter_num += 1
            for vec in X:
                for i in range(N):
                    for j in range(N):
                        v = weights[i, j] + vec[j]
                        weights[i, j] = weights[i, j] + self.learning_rate * v * (vec[i] - v * weights[i][j])

        return weights

    def reconstruct_sync(self, x):

        x = x.copy()
        self.previous_steps = [x]
        for i in range(0, self.max_iter):
            new_x = np.dot(self.Weights, x)
            new_x = self.activation(new_x)

            if np.array_equal(new_x, x):
                return new_x

            if any(np.array_equal(v, new_x) for v in self.previous_steps):
                print("Cycle detected!")
                self.previous_steps.append(new_x)
                return new_x

            self.previous_steps.append(new_x)

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

    def __make_diagonal_zero(self):
        for i in range(0, len(self.Weights)):
            self.Weights[i][i] = 0

    def __ojaRule3(self, X, N, M):

        Weights = np.zeros((N, N))
        Weights_copy = np.ones((N, N))

        while np.linalg.norm(Weights - Weights_copy) >= self.epsilon:
            Weights_copy = Weights.copy()

            for k in range(0, len(X)):
                x = np.transpose([X[k]])
                y = x if k == 0 else self.activation(np.dot(Weights, x))
                for i in range(0, N):
                    for j in range(0, N):
                        # if i == j:
                        #     Weights[i][i] = 0
                        #     continue
                        Weights[i][j] += self.learning_rate * y[i] * (x[j] - y[i] * Weights[i][j])
                        Weights[j][i] = Weights[i][j]
        return Weights
