import matplotlib.pyplot as plt
import numpy as np
import src.reader as reader
from src.Hopfield import HopfieldNetwork


def print_pcit(p,n,m):
    pict = []
    for i in range(0, n * m, m):
        pict.append(p[i:i+m])
    plt.imshow(pict)
    plt.show()


path = '../data/small-7x7.csv'
X, n, m = reader.read_data(path)

test = [1, -1, -1,	-1,	1,	-1,	1,	1,	1,	-1,	1,	-1,	1,	1,	-1,	-1,	1,	-1,	1,	-1,	-1,	-1,	-1,	-1,	1,	-1,	-1,	-1,	-1,	-1,	1,	-1,	1,	-1,	-1,	-1,	1,	-1,	-1,	-1,	1,	-1,	1,	-1,	-1,	-1,	-1,	-1,	1]
test = np.array(test)


network = HopfieldNetwork(0.04, 100)

network.train(X)

print_pcit(test, n, m)
pred = network.reconstruct(test)
print_pcit(pred, n, m)
