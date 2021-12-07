import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def plot_error_curve(errors, title=None):
    k = np.arange(1, len(errors) + 1)
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yscale('log')
    ax.grid()

    if title is not None:
        plt.title(title)
    plt.xlabel("k")
    plt.ylabel("Erro")
    ax.plot(k, errors, marker="o", markersize=5)

    plt.show()
