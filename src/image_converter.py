from PIL import Image
import os

def __convert_to_binary_arr(img_pixels):

    binary_arr = []
    for px in img_pixels:
        val = (px[0] + px[1] + px[2]) / 3
        if val < 128:
            binary_arr.append(-1)
        else:
            binary_arr.append(1)

    return binary_arr

def convert_images(folder_path, set_name):

    images_paths = []

    for file_path in os.listdir(folder_path):
        if ".jpg" in file_path:
            images_paths.append(folder_path + "/" + file_path)

    width = 0
    height = 0
    output_file_content = []
    for img_path in images_paths:
        im = Image.open(img_path)
        im = im.convert()
        pixels = list(im.getdata())
        width, height = im.size
        output_file_content.append(__convert_to_binary_arr(pixels))

    file_name = set_name + "-" + str(height) + "x" + str(width) + ".csv"

    with open(file_name, "w+") as output_file:

        for line in output_file_content:
            for i in range(len(line)):
                output_file.write(str(line[i]))
                if i < len(line) - 1:
                    output_file.write(",")

            output_file.write("\n")


convert_images("../img_s", "../data/buses_s")