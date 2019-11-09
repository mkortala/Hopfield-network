import numpy as np

class HopfieldNetwork:

    def __init__(self, learning_rate, max_iter):
        self.Weights = []
        self.activation = lambda x: np.sign(x)
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def train(self, X):
        self.Weights = self.hebbianRule(X, X.shape[1], X.shape[0])

    def hebbianRule(self, X, N, M):
        dot = np.dot(X.T, X)
        return 1/N * (dot - M * np.identity(N))


    def reconstruct(self, x):
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
