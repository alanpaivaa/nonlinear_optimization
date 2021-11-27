import numpy as np
from unconstrained_optimization.function import Function
from unconstrained_optimization.line_search import ConstantLineSearch, BacktrackingLineSearch, ExactLineSearch
from unconstrained_optimization.descent_methods import GradientDescent
from unconstrained_optimization.descent_methods import SteepestDescent, STEEPEST_DESCENT_NORM_EUCLIDEAN, STEEPEST_DESCENT_NORM_QUADRATIC, STEEPEST_DESCENT_NORM_L1
from unconstrained_optimization.descent_methods import NewtonStep
from unconstrained_optimization.plot_helper import plot_error_curve, plot_levels

# Define starting point for the optimization
x0 = np.array([1, 1])
epsilon = 1e-7

# Define function
f = Function()

# Define line search method
# line_search = ConstantLineSearch(t=0.01)
# line_search = BacktrackingLineSearch(f=f, alpha=0.3, beta=0.7)
# line_search = ExactLineSearch(f=f)

# Gradient Descent
# optimizer = GradientDescent(f=f, line_search=line_search, epsilon=epsilon)

# Steepest Descent
# p = np.array([[5, 0], [0, 5]])  # Cond = 1
# p = np.array([[5, 0], [0, 10]])  # Cond = 2
# p = np.array([[5, 0], [0, 100]])  # Cond = 20
# optimizer = SteepestDescent(f=f, line_search=line_search, p=p, norm=STEEPEST_DESCENT_NORM_L1, epsilon=epsilon)

# Newton Step
optimizer = NewtonStep(f=f, alpha=0.1, beta=0.7, epsilon=epsilon)

# Optimization
iterations, x_min, path, errors = optimizer.optimize(x0)

# Print results
print("Número de Iterações: %d" % iterations)
print("x* = [%.10f, %.10f]" % (x_min[0], x_min[1]))
print("f(x*) = %.10f" % f(x_min))

# Plot error curve
plot_error_curve(errors=errors)

# Plot levels
plot_levels(f, path)

print("Done!")
