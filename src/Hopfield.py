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
            self.Weights = self.__ojaRule(X, X.shape[1], X.shape[0])

    def __hebbianRule(self, X, N, M):
        dot = np.dot(X.T, X)
        return 1 / N * (dot - M * np.identity(N))

    def __ojaRule(self, X, N, M):

        W = self.__hebbianRule(X, N, M)
        # W = np.random.normal(scale=0.25, size=(N, N))
        Weights_prev = np.zeros((N, N))
        z = 0
        while np.linalg.norm(W - Weights_prev) > self.epsilon:
            Weights_prev = W.copy()
            #y = np.sum(np.dot(self.Weights, X.T), axis=1).reshape((N, 1))
            for i in range(0, len(X)):
                y = np.dot(W, X[i]).reshape((N, 1))
                W += self.learning_rate * (y * X[i] - np.square(y) * W)
                W = self.__make_diagonal_0(W)
                # x = X[i]
                # for j in range(0, N):
                #     for k in range(0, N):
                #         if j == k:
                #             continue
                #         W[j, k] += self.learning_rate * (y[j] * x[k] - np.square(y[j]) * W[j, k])
            Energy = -1/2 * np.sum(W * np.dot(X.T, X))
            print('Energy: ', Energy, z)
            z += 1
        print(np.sqrt(np.sum(W*W, axis=1)))
        return W

    def reconstruct_sync(self, x):
        x = x.copy()
        self.previous_steps = [x]
        for i in range(0, self.max_iter):
            new_x = np.dot(self.Weights, x)
            new_x = self.activation(new_x)

            self.previous_steps.append(new_x)
            if np.array_equal(new_x, x):
                return new_x

            # if any(np.array_equal(v, new_x) for v in self.previous_steps):
            #     print("Cycle detected!")
            #     return new_x

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

    def __make_diagonal_0(self, Z):
        for i in range(0, len(Z)):
            Z[i][i] = 0
        return Z
