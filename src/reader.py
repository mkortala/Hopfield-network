import numpy as np


def read_data(filename):
    f = filename.split(sep='.')[2]
    f = f.split(sep='-')[1]
    [n, m] = f.split(sep='x')
    n = int(n)
    m = int(m)
    with open(filename) as file:
        return read_from_file(file), n, m


def read_from_file(file):
    X = []
    for line in file:
        row = line.split(sep=',')
        row = [float(elem) for elem in row]
        X.append(row)

    return np.array(X)
