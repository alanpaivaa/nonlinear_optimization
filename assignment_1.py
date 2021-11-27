import numpy as np
from unconstrained_optimization.function import Function
from unconstrained_optimization.line_search import ConstantLineSearch, BacktrackingLineSearch, ExactLineSearch
from unconstrained_optimization.descent_methods import GradientDescent
from unconstrained_optimization.descent_methods import SteepestDescent, STEEPEST_DESCENT_NORM_EUCLIDEAN, STEEPEST_DESCENT_NORM_QUADRATIC, STEEPEST_DESCENT_NORM_L1

# TODO: Remove seed
# TODO: Grid Search for Hyper params

# Define starting point for the optimization
np.random.seed(11)
x0 = np.random.uniform(low=-4, high=0, size=(2,))

# Define function
f = Function()

# Define line search method
# line_search = ConstantLineSearch(t=0.01)
line_search = BacktrackingLineSearch(f=f, alpha=0.3, beta=0.7)
# line_search = ExactLineSearch(f=f)

# Define descent method
# optimizer = GradientDescent(f=f, line_search=line_search)

p = np.array([[5, 0], [0, 5]])  # Cond = 1
# p = np.array([[5, 0], [0, 10]])  # Cond = 2
# p = np.array([[5, 0], [0, 10000]])  # Cond = 2000
optimizer = SteepestDescent(f=f, line_search=line_search, p=p, norm=STEEPEST_DESCENT_NORM_L1)

# Optimization
iterations, x_min = optimizer.optimize(x0)

# Print results
print("Número de Iterações: %d" % iterations)
print("x* = [%.10f, %.10f]" % (x_min[0], x_min[1]))
print("f(x*) = %.10f" % f(x_min))
