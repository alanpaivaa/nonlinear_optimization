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


def plot_levels(f, path):
    x_space = np.linspace(-1.5, 1.3, 100)
    y_space = np.linspace(-1.5, 1.2, 100)
    X, Y = np.meshgrid(x_space, y_space)

    Z = list()
    for x, y in zip(X, Y):
        row = list()
        for i in range(len(x)):
            row.append(f(np.array([x[i], y[i]])))
        Z.append(row)
    Z = np.array(Z)

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X, Y, Z, levels=25)
    fig.colorbar(cp)
    ax.set_title('Curvas de Nível')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.plot(path[:, 0], path[:, 1], color='white', marker="o", markersize=5)
    plt.show()
