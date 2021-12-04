import numpy as np
from constrained_optimization.function import Function
from constrained_optimization.descent_methods import NewtonStepEqualConstraints

a = np.load('constrained_optimization/a.npy')
x_hat = np.load('constrained_optimization/x_hat.npy')
b = a @ x_hat


f = Function()
optimizer = NewtonStepEqualConstraints(f=f, a=a, b=b, alpha=0.1, beta=0.7, epsilon=1e-20)
iterations, x_min, path, errors = optimizer.optimize(x_hat)
print(f(x_min))
print(len(errors))
print(np.unique(np.isclose(a @ x_min, b)).item())
print(np.unique(np.isclose(x_min, x_hat)).item())
