import numpy as np
from constrained_optimization.line_search import BacktrackingLineSearch


class NewtonStepEqualConstraints:
    def __init__(self, f, a, b, alpha, beta, epsilon):
        self.f = f
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.line_search = BacktrackingLineSearch(f, alpha, beta)

    def get_delta_x(self, x):
        # Calculate sizes
        n = x.shape[0]
        p = self.a.shape[0]

        # Make matrix of matrices
        lhs = np.zeros((n + p, n + p))
        lhs[:n, :n] = self.f.hessian(x)  # Top left is the hessian
        lhs[n:, :n] = self.a             # Bottom left is A
        lhs[:n, n:] = self.a.T           # Bottom right is A transposed

        # Make result matrix
        rhs = np.zeros((n + p, 1))
        rhs[:n, 0] = -self.f.gradient(x)  # Top is minus gradient

        # Calculate v and w
        res = np.linalg.inv(lhs) @ rhs
        v = res[:n][:].T[0]  # Top is v
        # w = res[n:][:].T[0]  # Bottom is w

        return v

    def get_decrement(self, x, delta_x):
        grad = self.f.gradient(x).reshape((x.shape[0], -1))
        lambda2 = -grad.T @ delta_x.reshape((x.shape[0], -1))
        return lambda2.item() / 2

    def optimize(self, x):
        i = 0
        x_min = x
        errors = []
        path = [x_min]

        while True:
            i += 1

            delta_x = self.get_delta_x(x_min)
            decrement = self.get_decrement(x_min, delta_x)
            errors.append(decrement)

            if decrement < self.epsilon:
                return i, x_min, np.array(path), np.array(errors)

            t = self.line_search(x_min, delta_x)
            x_min = x_min + t * delta_x

            # Should be a feasible point
            assert np.unique(np.isclose(self.a @ x_min, self.b)).item()

            path.append(x_min)
