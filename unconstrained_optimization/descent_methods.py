import numpy as np
from unconstrained_optimization.line_search import BacktrackingLineSearch


class AbstractDescent:
    def __init__(self, f, line_search, epsilon):
        self.f = f
        self.line_search = line_search
        self.epsilon = epsilon

    def get_delta_x(self, x):
        return None

    def optimize(self, x):
        converged = False
        i = 0
        min_x = x
        errors = []
        path = [min_x]
        while not converged:
            delta_x = self.get_delta_x(min_x)
            t = self.line_search(min_x, delta_x)
            min_x = min_x + t * delta_x
            path.append(min_x)
            error = np.linalg.norm(delta_x, ord=2)
            errors.append(error)
            converged = error < self.epsilon
            i += 1
            # print(i, min_x, self.f(min_x))
        return i, min_x, np.array(path), np.array(errors)


class GradientDescent(AbstractDescent):
    def get_delta_x(self, x):
        return -self.f.gradient(x)


# Steepest Descent Norm types
STEEPEST_DESCENT_NORM_EUCLIDEAN = 'st_norm_euclidean'
STEEPEST_DESCENT_NORM_QUADRATIC = 'st_norm_quadratic'
STEEPEST_DESCENT_NORM_L1 = 'st_norm_l1'


class SteepestDescent(AbstractDescent):
    def __init__(self, f, line_search, p, norm, epsilon):
        super().__init__(f=f, line_search=line_search, epsilon=epsilon)
        self.norm = norm
        self.p = p

    def get_delta_x_euclidean_norm(self, x):
        return -self.f.gradient(x)

    def get_delta_x_quadratic_norm(self, x):
        return -np.linalg.inv(self.p) @ self.f.gradient(x)

    def get_delta_x_l1_norm(self, x):
        grad = self.f.gradient(x)

        # Get the ith index
        inf_norm = np.linalg.norm(grad, ord=np.inf)
        i = np.argwhere(np.abs(grad) == inf_norm).item()

        # Get the standard basis vector
        e_i = np.zeros(x.shape[0])
        e_i[i] = 1

        return -grad[i] * e_i

    def get_delta_x(self, x):
        if self.norm == STEEPEST_DESCENT_NORM_EUCLIDEAN:
            return self.get_delta_x_euclidean_norm(x)
        elif self.norm == STEEPEST_DESCENT_NORM_QUADRATIC:
            return self.get_delta_x_quadratic_norm(x)
        elif self.norm == STEEPEST_DESCENT_NORM_L1:
            return self.get_delta_x_l1_norm(x)
        else:
            raise Exception("Invalid norm")


class NewtonStep(AbstractDescent):
    def __init__(self, f, alpha, beta, epsilon):
        line_search = BacktrackingLineSearch(f, alpha, beta)
        super().__init__(f, line_search, epsilon)

    def get_delta_x(self, x):
        grad = self.f.gradient(x).reshape((x.shape[0], -1))
        return -np.linalg.inv(self.f.hessian(x)) @ grad

    def get_decrement(self, x):
        grad = self.f.gradient(x).reshape((x.shape[0], -1))
        lambda2 = grad.T @ np.linalg.inv(self.f.hessian(x)) @ grad
        return lambda2.item() / 2

    def optimize(self, x):
        i = 0
        min_x = x
        errors = []
        path = [min_x]
        while True:
            decrement = self.get_decrement(min_x)
            errors.append(decrement)
            if decrement < self.epsilon:
                return i, min_x, np.array(path), np.array(errors)
            i += 1
            delta_x = self.get_delta_x(min_x)
            delta_x = delta_x.T[0]
            t = self.line_search(min_x, delta_x)
            min_x = min_x + t * delta_x
            path.append(min_x)
