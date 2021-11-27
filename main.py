import numpy as np
from function import Function
from line_search import ExactLineSearch, BacktrackingLineSearch, ConstantLineSearch
from descent_methods import GradientDescent

# TODO: Remove seed
np.random.seed(11)
function = Function()

# TODO: Grid Search for Hyper params
# line_search = ConstantLineSearch(t=0.01)
line_search = BacktrackingLineSearch(f=function, alpha=0.3, beta=0.7)
# line_search = ExactLineSearch(f=function)

descent_method = GradientDescent(f=function, line_search=line_search)

x0 = np.random.uniform(low=-4, high=0, size=(2,))
iterations, x_min = descent_method(x0)
print("Número de Iterações: %d" % iterations)
print("x* = [%.10f, %.10f]" % (x_min[0], x_min[1]))
print("f(x*) = %.10f" % function(x_min))
