import numpy as np
import src.reader as reader
import src.presentation as pres
import src.data_disturber as dis
from src.Hopfield import HopfieldNetwork, LearningType
import src.experiments as exp

print("Reading data file...")

#path = '../data/letters_abc-8x12.csv'
#path = '../data/animals-14x9.csv'
path = '../data/small-7x7.csv'
#path = '../data/test-3x3.csv'
#path = '../data/large-25x25.csv'

X, n, m = reader.read_data(path)

print("Displaying file data...")
pres.print_dataset(X, n, m, path)

curr_image_idx = 0
curr_image = X[curr_image_idx].copy()

print("Creating network...")
learning_rate = 1.0 / len(curr_image)
network = HopfieldNetwork(learning_rate, 1000, LearningType.Hebbian, 1e-10)
network.train(X)

#exp.single_test(curr_image, network, n, m)
exp.multiple_tests(X, network, n, m)