import numpy as np


class ExactLineSearch:
    def __init__(self, f):
        self.f = f

    def __call__(self, x, delta_x):
        lin_space = np.linspace(0, 1, 100000)
        res = np.array([self.f(x + s * delta_x) for s in lin_space])
        min_arg = np.argmin(res)
        return lin_space[min_arg]


class BacktrackingLineSearch:
    def __init__(self, f, alpha, beta):
        self.f = f
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x, delta_x):
        t = 1
        while self.f(x + t * delta_x) > self.f(x) + self.alpha * t * (self.f.gradient(x) @ delta_x.reshape(2, 1)):
            t = self.beta * t
        return t
