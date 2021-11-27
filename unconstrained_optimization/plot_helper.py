import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


def plot_surface(f):
    x1 = list()
    x2 = list()
    for a in np.linspace(-4, 2, 800):
        for b in np.linspace(-4, 2, 800):
            x1.append(a)
            x2.append(b)
    x1 = np.array(x1)
    x2 = np.array(x2)
    y = f(np.array([x1, x2]))

    ax = plt.axes(projection="3d")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.plot3D(x1, x2, y)
    plt.show()


def plot_error_curve(errors):
    k = np.arange(1, len(errors) + 1)
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yscale('log')
    ax.grid()

    plt.title("Curva de erros")
    plt.xlabel("k")
    plt.ylabel("Erro")
    ax.plot(k, errors, marker="o", markersize=5)

    plt.show()
