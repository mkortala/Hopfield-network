import numpy as np
import src.reader as reader
import src.presentation as pres
import src.data_disturber as dis
from src.Hopfield import HopfieldNetwork, LearningType

network_async_mode = False

print("Reading data file...")
#path = '../data/letters_abc-8x12.csv'
path = '../data/small-7x7.csv'
X, n, m = reader.read_data(path)

print("Displaying file data...")
pres.print_dataset(X, n, m, path)

curr_image_idx = 2
curr_image = X[curr_image_idx].copy()

print("Disturbing image...")
test_image = dis.disturb_image(curr_image, 0)

print("Creating network...")
network = HopfieldNetwork(0.001, 1000, LearningType.Ojas, 1e-10)
network.train(X)

print("Prediction...")
if network_async_mode:
    pred_image = network.reconstruct_async(test_image)
else:
    pred_image = network.reconstruct_sync(test_image)

print("Displaying predicted image...")
pres.print_experiment_result(X[curr_image_idx], test_image, pred_image, n, m)

print("Displaying construction steps...")
pres.print_reconstruction_steps(network.previous_steps, n, m)
