import numpy as np


class AbstractDescent:
    def __init__(self, f, line_search, epsilon=1e-5):
        self.f = f
        self.line_search = line_search
        self.epsilon = epsilon

    def get_delta_x(self, x):
        return None

    def optimize(self, x):
        converged = False
        i = 0
        min_x = x
        while not converged:
            delta_x = self.get_delta_x(min_x)
            t = self.line_search(min_x, delta_x)
            min_x = min_x + t * delta_x
            converged = np.linalg.norm(delta_x, ord=2) < self.epsilon
            i += 1
            # print(i, min_x, self.f(min_x))
        return i, min_x


class GradientDescent(AbstractDescent):
    def get_delta_x(self, x):
        return -self.f.gradient(x)


# Steepest Descent Norm types
STEEPEST_DESCENT_NORM_EUCLIDEAN = 'st_norm_euclidean'
STEEPEST_DESCENT_NORM_QUADRATIC = 'st_norm_quadratic'


class SteepestDescent(AbstractDescent):
    def __init__(self, f, line_search, p, norm, epsilon=1e-5):
        super().__init__(f=f, line_search=line_search, epsilon=epsilon)
        self.norm = norm
        self.p = p

    def get_delta_x_euclidean_norm(self, x):
        return -self.f.gradient(x)

    def get_delta_x_quadratic_norm(self, x):
        return -np.linalg.inv(self.p) @ self.f.gradient(x)

    def get_delta_x(self, x):
        if self.norm == STEEPEST_DESCENT_NORM_EUCLIDEAN:
            return self.get_delta_x_euclidean_norm(x)
        elif self.norm == STEEPEST_DESCENT_NORM_QUADRATIC:
            return self.get_delta_x_quadratic_norm(x)
        else:
            raise Exception("Invalid norm")
