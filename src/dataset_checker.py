import numpy as np


def check_data_set(X, network):
    """Bada warunek poprawnego zapamietywania i rozpoznawania sieci Hopfielda (Tw. 3.4 z ksiazki prof. Mandziuka)
    """

    m = len(X)
    n = len(X[0])

    for k in range(m):
        for i in range(n):
            sum = 0
            for j in range(n):
                sum += network.Weights[i, j] * X[k][j]
            sum *= X[k][i]

            if sum <= 0:
                return False

    return True
