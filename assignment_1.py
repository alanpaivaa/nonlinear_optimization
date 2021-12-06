import numpy as np
import time
from unconstrained_optimization.function import Function
from unconstrained_optimization.line_search import BacktrackingLineSearch, ExactLineSearch
from unconstrained_optimization.descent_methods import GradientDescent
from unconstrained_optimization.descent_methods import SteepestDescent, STEEPEST_DESCENT_NORM_EUCLIDEAN, \
    STEEPEST_DESCENT_NORM_QUADRATIC, STEEPEST_DESCENT_NORM_L1
from unconstrained_optimization.descent_methods import NewtonStep
from unconstrained_optimization.plot_helper import plot_error_curve, plot_levels


def get_gradient_descent_optimizer(f):
    epsilon = 1e-7
    # line_search = ExactLineSearch(f=f)
    line_search = BacktrackingLineSearch(f=f, alpha=0.124490, beta=0.387640)
    return GradientDescent(f=f, line_search=line_search, epsilon=epsilon)


def get_steepest_descent_optimizer(f):
    epsilon = 1e-7

    # Euclidean setup
    # return SteepestDescent(f=f,
    #                        line_search=BacktrackingLineSearch(f=f, alpha=0.124490, beta=0.387640),
    #                        p=None,
    #                        norm=STEEPEST_DESCENT_NORM_EUCLIDEAN,
    #                        epsilon=epsilon)

    # Quadratic with P1 setup
    # return SteepestDescent(f=f,
    #                        line_search=BacktrackingLineSearch(f=f, alpha=0.1, beta=0.585393),
    #                        p=np.array([[5, 0], [0, 5]]),
    #                        norm=STEEPEST_DESCENT_NORM_QUADRATIC,
    #                        epsilon=epsilon)

    # Quadratic with P2 setup
    # return SteepestDescent(f=f,
    #                        line_search=BacktrackingLineSearch(f=f, alpha=0.108163, beta=0.684270),
    #                        p=np.array([[5, 0], [0, 10]]),
    #                        norm=STEEPEST_DESCENT_NORM_QUADRATIC,
    #                        epsilon=epsilon)

    # Quadratic with P3 setup
    # return SteepestDescent(f=f,
    #                        line_search=BacktrackingLineSearch(f=f, alpha=0.1, beta=0.423596),
    #                        p=np.array([[5, 0], [0, 100]]),
    #                        norm=STEEPEST_DESCENT_NORM_QUADRATIC,
    #                        epsilon=epsilon)

    # L1 setup
    return SteepestDescent(f=f,
                           line_search=BacktrackingLineSearch(f=f, alpha=0.402041, beta=0.801124),
                           p=None,
                           norm=STEEPEST_DESCENT_NORM_L1,
                           epsilon=epsilon)


def get_newton_step_optimizer(f):
    return NewtonStep(f=f, alpha=0.1, beta=0.1, epsilon=1e-20)


def main():
    # Define function
    f = Function()

    # Define starting point for the optimization
    x_start = np.array([1, 1])

    # optimizer = get_gradient_descent_optimizer(f)  # Gradient Descent
    # optimizer = get_steepest_descent_optimizer(f)  # Steepest Descent
    optimizer = get_newton_step_optimizer(f)       # Newton Step

    # Optimization
    start_time = time.time()
    iterations, x_min, path, errors = optimizer.optimize(x_start)
    end_time = time.time()

    # Print results
    print("Número de Iterações: %d" % iterations)
    print("x* = [%.10f, %.10f]" % (x_min[0], x_min[1]))
    print("f(x*) = %.10f" % f(x_min))
    print("Tempo de execução: %d ms" % ((end_time - start_time) * 1000))

    # Plot error curve
    plot_error_curve(errors=errors)

    # Plot levels
    plot_levels(f, path)


main()
