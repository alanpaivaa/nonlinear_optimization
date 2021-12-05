import numpy as np
from constrained_optimization.line_search import BacktrackingLineSearch


class FeasibleStartNewtonStep:
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
        lhs[n:, :n] = self.a  # Bottom left is A
        lhs[:n, n:] = self.a.T  # Bottom right is A transposed

        # Make result matrix
        rhs = np.zeros((n + p, 1))
        rhs[:n, 0] = -self.f.gradient(x)  # Top is minus gradient
        rhs[n:, 0] = -(self.a @ x - self.b)

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
            assert np.allclose(self.a @ x_min, self.b)

            path.append(x_min)


class InfeasibleStartNewtonStep:
    def __init__(self, f, a, b, alpha, beta, epsilon):
        self.f = f
        self.a = a
        self.b = b
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def get_deltas(self, x, nu):
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
        rhs[n:, 0] = -(self.a @ x - self.b)

        # Calculate v and w
        res = np.linalg.inv(lhs) @ rhs
        delta_x = res[:n][:].T[0]  # Top is v
        w = res[n:][:].T[0]        # Bottom is w
        delta_nu = w - nu

        return delta_x, delta_nu

    def get_r(self, x, nu):
        r_primal = self.a @ x - self.b  # (p,1)
        r_dual = self.f.gradient(x) + self.a.T @ nu  # (n,1)
        return r_dual, r_primal

    @staticmethod
    def get_r_norm(r):
        r_dual, r_primal = r
        norm_dual = np.linalg.norm(r_dual, ord=2)
        norm_primal = np.linalg.norm(r_primal, ord=2)
        return np.array([norm_dual, norm_primal])

    def get_t(self, x, delta_x, nu, delta_nu):
        t = 1.0
        while True:
            # Avoid leaving the domain of the function
            if not self.f.is_in_domain(x + t * delta_x):
                t = self.beta * t
                continue

            # Get primal and residual values
            r = self.get_r(x, nu)
            delta_r = self.get_r(x + t * delta_x, nu + t * delta_nu)

            if not np.all(self.get_r_norm(delta_r) > (1 - self.alpha * t) * self.get_r_norm(r)):
                return t

            t = self.beta * t

    def optimize(self, x):
        i = 0
        x_min = x
        nu_min = np.ones(self.a.shape[0])
        errors = []
        path = [x_min]

        while True:
            i += 1

            # Compute primal and dual steps
            delta_x, delta_nu = self.get_deltas(x_min, nu_min)

            # Backtracking line search
            t = self.get_t(x_min, delta_x, nu_min, delta_nu)

            # Update primal and dual variables
            x_min = x_min + t * delta_x
            nu_min = nu_min + t * delta_nu
            path.append(x_min)

            # Calculate error
            error = self.get_r_norm(self.get_r(x_min, nu_min))
            errors.append(error)

            # Verify feasibility of the minimum
            is_feasible = np.allclose(self.a @ x_min, self.b)

            # Finish when point is feasible and norm of r is less than the tolerance
            if is_feasible and np.all(error <= self.epsilon):
                return i, x_min, np.array(path), np.array(errors)
