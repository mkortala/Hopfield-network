import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import numpy as np
import math

def print_pcit(p, n, m):

    pict = []
    for i in range(0, n * m, m):
        pict.append(p[i : i + m])
    plt.imshow(pict)
    plt.show()

def print_experiment_result(p_orig, p_test, p_found, n, m):

    fig, axs = plt.subplots(1, 3)

    pict0 = []
    for i in range(0, n * m, m):
        pict0.append(p_orig[i: i + m])
    axs[0].set_title("Original image")
    axs[0].imshow(pict0)

    pict1 = []
    for i in range(0, n * m, m):
        pict1.append(p_test[i: i + m])
    axs[1].set_title("Test image")
    axs[1].imshow(pict1)

    pict2 = []
    for i in range(0, n * m, m):
        pict2.append(p_found[i: i + m])

    axs[2].set_title("Found image")
    axs[2].imshow(pict2)

    plot_title = 'Result'
    if np.array_equal(p_orig, p_found):
        plot_title += " (OK)"

    fig.suptitle(plot_title, fontsize=15)
    plt.show()

def print_reconstruction_steps(steps, n, m):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)

    pict = []
    for i in range(0, n * m, m):
        pict.append(steps[0][i: i + m])
    plt.title("Reconstruction steps")
    im = plt.imshow(pict)

    def __update_by_sider(val):
        pict = []
        for i in range(0, n * m, m):
            pict.append(steps[int(val)][i: i + m])
        im.set_data(pict)
        fig.canvas.draw_idle()

    def __next(e):
        if steps_slider.val < len(steps) - 1:
            steps_slider.set_val(steps_slider.val + 1)
    def __prev(e):
        if steps_slider.val > 0:
            steps_slider.set_val(steps_slider.val - 1)

    axcolor = "lightgoldenrodyellow"
    axsteps = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    steps_slider = Slider(axsteps, "Step ", 0, len(steps) - 1, valinit=0, valstep=1, valfmt="%1.0f")
    steps_slider.on_changed(__update_by_sider)

    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(__next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(__prev)

    plt.show()


def print_dataset(X, n, m, plot_title):

    plot_columns = 3
    plot_rows = X.shape[0] / plot_columns
    if X.shape[0] % plot_columns != 0:
        plot_rows += 1

    fig, axs = plt.subplots(int(plot_rows), int(plot_columns))

    for img_num in range(X.shape[0]):

        pict = []
        p = X[img_num]
        for i in range(0, n * m, m):
            pict.append(p[i : i + m])

        if int(plot_rows) == 1:
            axs[int(img_num % plot_columns)].imshow(pict)
            axs[int(img_num % plot_columns)].set_title(str(img_num))
        else:
            axs[int(img_num / plot_columns), int(img_num % plot_columns)].imshow(pict)
            axs[int(img_num / plot_columns), int(img_num % plot_columns)].set_title(str(img_num))

    fig.suptitle(plot_title, fontsize=15)

    plt.show()