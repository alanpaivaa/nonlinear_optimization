import matplotlib.pyplot as plt
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
