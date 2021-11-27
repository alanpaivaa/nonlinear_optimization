import numpy as np


class AbstractDescent:
    def __init__(self, f, line_search, epsilon=1e-5):
        self.f = f
        self.line_search = line_search
        self.epsilon = epsilon

    def get_delta_x(self, x):
        return -self.f.gradient(x)[0]


class GradientDescent(AbstractDescent):
    def optimize(self, x):
        converged = False
        i = 0
        min_x = x
        while not converged:
            delta_x = self.get_delta_x(min_x)
            t = self.line_search(min_x, delta_x)
            min_x = min_x + t * delta_x
            converged = np.linalg.norm(delta_x) < self.epsilon
            i += 1
            # print(i, min_x, self.f(min_x))
        return i, min_x
