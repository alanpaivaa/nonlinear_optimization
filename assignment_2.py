import numpy as np
from constrained_optimization.function import Function
from constrained_optimization.descent_methods import FeasibleStartNewtonStep, InfeasibleStartNewtonStep

a = np.load('constrained_optimization/a.npy')
x_hat = np.load('constrained_optimization/x_hat.npy')
b = a @ x_hat

# Hyper params
alpha = 0.1
beta = 0.7


f = Function()

# Feasible starting point
# x0 = x_hat
# epsilon = 1e-20
# optimizer = FeasibleStartNewtonStep(f=f, a=a, b=b, alpha=alpha, beta=beta, epsilon=epsilon)

# Infeasible starting point
x0 = np.ones(x_hat.shape)
epsilon = 1e-12
optimizer = InfeasibleStartNewtonStep(f=f, a=a, b=b, alpha=alpha, beta=beta, epsilon=epsilon)

iterations, x_min, path, errors = optimizer.optimize(x0)

print(f(x_min))
print(len(errors))
print(np.allclose(a @ x_min, b))
print(np.allclose(x_min, x0))
