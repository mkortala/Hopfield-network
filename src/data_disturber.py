import random


def disturb_image(image, amount_pixels):

    pixels_to_change = []

    for px in range(amount_pixels):
        idx = random.randint(0, len(image) - 1)
        while idx in pixels_to_change:
            idx = (idx + 1) % len(image)

        pixels_to_change.append(idx)

    for px in pixels_to_change:
        image[px] = (-1) * image[px]

    return image
