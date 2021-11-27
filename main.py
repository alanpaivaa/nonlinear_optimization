import numpy as np
import matplotlib.pyplot as plt


def plot_surface(x1, x2, y):
    ax = plt.axes(projection="3d")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("y")
    ax.plot3D(x1, x2, y)
    plt.show()


# x1 = list()
# x2 = list()
# for a in np.linspace(-4, 2, 800):
#     for b in np.linspace(-4, 2, 800):
#         x1.append(a)
#         x2.append(b)
# x1 = np.array(x1)
# x2 = np.array(x2)
# y = np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 2*x2 - 0.1) + np.exp(-x1 - 0.2)

# Analytical
# argmin = np.argmin(y)
# print(x1[argmin], x2[argmin])
# Correct values: -0.38798498122653324 -0.08010012515644549
# plot_surface(x1, x2, y)

def f(x_k):
    x1 = x_k[0]
    x2 = x_k[1]
    return np.exp(x1 + 3 * x2 - 0.1) + np.exp(x1 - 2 * x2 - 0.1) + np.exp(-x1 - 0.2)


def eq_one(x_k):
    x1 = x_k[0]
    x2 = x_k[1]
    return np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 2*x2 - 0.1) - np.exp(-x1 - 0.2)


def eq_two(x_k):
    x1 = x_k[0]
    x2 = x_k[1]
    return np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 2*x2 - 0.1) + np.exp(-x1 - 0.2)


def eq_three(x_k):
    x1 = x_k[0]
    x2 = x_k[1]
    return 3 * np.exp(x1 + 3*x2 - 0.1) - 2*np.exp(x1 - 2*x2 - 0.1)


def eq_four(x_k):
    x1 = x_k[0]
    x2 = x_k[1]
    return 9 * np.exp(x1 + 3*x2 - 0.1) + 4 * np.exp(x1 - 2*x2 - 0.1)


def gradient(x_k):
    return np.array([eq_one(x_k), eq_three(x_k)]).reshape(1, 2)


def hessian(x_k):
    return np.array([
        [eq_two(x_k), eq_three(x_k)],
        [eq_three(x_k), eq_four(x_k)]
    ])


def get_backtracking_t(alpha, beta, x_k, delta_x):
    t = 1
    while f(x_k + t * delta_x) > f(x_k) + alpha * t * (gradient(x_k) @ delta_x.reshape(2, 1)):
        t = beta * t
    return t


def get_exact_line_t(x_k, delta_x):
    space = np.linspace(0, 1, 1000)
    res = np.array([f(x_k + s * delta_x) for s in space])
    argmin = np.argmin(res)
    return space[argmin]


def gradient_descent(x_k):
    epsilon = 1e-5
    converged = False
    i = 0
    while not converged:
        delta_x = -gradient(x_k)[0]
        # TODO: Grid Search
        # t = 0.01
        t = get_backtracking_t(0.3, 0.7, x_k, delta_x)
        # t = get_exact_line_t(x_k, delta_x)
        x_k = x_k + t * delta_x
        converged = np.linalg.norm(delta_x) < epsilon
        i += 1
        print(i, x_k, f(x_k))
    return x_k


np.random.seed(11)
x0 = np.random.uniform(low=-4, high=0, size=(2,))
x_min = gradient_descent(x0)
print(x_min)
