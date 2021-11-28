import numpy as np
from constrained_optimization.function import Function

a = np.load('constrained_optimization/a.npy')
x_hat = np.load('constrained_optimization/x_hat.npy')
b = a @ x_hat


# f = Function()
# print(f(np.array([-3])))
