import numpy as np
import random


def generate_image(n, m):

    image = np.zeros((n * m,))

    for i in range(n * m):
        r = random.randint(0, 1)
        if r == 0:
            image[i] = -1
        else:
            image[i] = 1

    return image
