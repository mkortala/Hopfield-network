import matplotlib.pyplot as plt
import math

def print_pcit(p, n, m):

    pict = []
    for i in range(0, n * m, m):
        pict.append(p[i : i + m])
    plt.imshow(pict)
    plt.show()

def print_experiment_result(p_test, p_found, n, m):

    fig, axs = plt.subplots(1, 2)

    pict1 = []
    for i in range(0, n * m, m):
        pict1.append(p_test[i: i + m])
    axs[0].set_title("Test image")
    axs[0].imshow(pict1)

    pict2 = []
    for i in range(0, n * m, m):
        pict2.append(p_found[i: i + m])

    axs[1].set_title("Found image")
    axs[1].imshow(pict2)

    fig.suptitle('Result', fontsize=15)
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

        if plot_rows == 1:
            axs[int(img_num % plot_columns)].imshow(pict)
        else:
            axs[int(img_num / plot_columns), int(img_num % plot_columns)].imshow(pict)

    fig.suptitle(plot_title, fontsize=15)

    plt.show()