import numpy as np
import src.presentation as pres
import src.data_disturber as dis
from src.Hopfield import HopfieldNetwork, LearningType

repetitions_count = 1
disturbed_pixels = 0

network_async_mode = True

def single_test(curr_image, network, n, m):

    print("Performing single test...")

    print("Disturbing image...")
    test_image = dis.disturb_image(curr_image.copy(), disturbed_pixels)

    print("Prediction...")
    if network_async_mode:
        pred_image = network.reconstruct_async(test_image)
    else:
        pred_image = network.reconstruct_sync(test_image)

    print("Displaying predicted image...")
    pres.print_experiment_result(curr_image, test_image, pred_image, n, m)

    print("Displaying construction steps...")
    pres.print_reconstruction_steps(network.previous_steps, n, m)


def multiple_tests(images_set, network, n, m):

    print("Performing multiple tests...")

    disturbed_images = []
    for img in images_set:
        dis_img = dis.disturb_image(img.copy(), disturbed_pixels)
        disturbed_images.append(dis_img)

    # wyniki eksperymentow dla poszczegolnych obrazow
    # (pozytywne, wszystkie)
    experiment_results = np.zeros((len(disturbed_images), 2))

    for i in range(len(disturbed_images)):

        for exp_num in range(repetitions_count):
            if network_async_mode:
                pred_image = network.reconstruct_async(disturbed_images[i])
            else:
                pred_image = network.reconstruct_sync(disturbed_images[i])

            experiment_results[i][1] += 1
            if np.array_equal(pred_image, images_set[i]):
                experiment_results[i][0] += 1

            pres.print_experiment_result(images_set[i], disturbed_images[i], pred_image, n, m)

    print("Experiments results:")
    for i in range(0, len(experiment_results)):
        exp_reult = (experiment_results[i][0] / experiment_results[i][1]) * 100.0
        print("{:4d} {:>3.2f} %".format(i, exp_reult))
