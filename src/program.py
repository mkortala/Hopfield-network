import numpy as np
import src.reader as reader
import src.presentation as pres
from src.Hopfield import HopfieldNetwork, LearningType
import src.experiments as exp
import src.dataset_checker as chk

print("Reading data file...")

# path = '../data/animals-14x9.csv'
path = '../data/large-25x25.csv'
# path = '../data/large-25x25.plus.csv'
# path = '../data/large-25x50.csv'
# path = '../data/letters-14x20.csv'
# path = '../data/letters_abc-8x12.csv'
# path = '../data/OCRA-12x30-cut.csv'
# path = '../data/small-7x7.csv'
# path = '../data/test-3x3.csv'

# path = '../data/buses-200x300.csv'

X, n, m = reader.read_data(path)

print("Displaying file data...")
pres.print_dataset(X, n, m, path)

curr_image_idx = 4
curr_image = X[curr_image_idx].copy()

print("Creating network...")
learning_rate = 1.0 / len(curr_image)
network = HopfieldNetwork(learning_rate, 1000, LearningType.Hebbian, 1)
network.train(X)

print("Checking network condition...")
if chk.check_data_set(X, network):
    print("Dataset OK!")
else:
    print("Dataset WRONG!")

# exp.single_test(curr_image, network, n, m)
exp.multiple_tests(X, network, n, m)