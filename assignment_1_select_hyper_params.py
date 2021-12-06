import numpy as np
from unconstrained_optimization.function import Function
from unconstrained_optimization.line_search import BacktrackingLineSearch
from unconstrained_optimization.descent_methods import GradientDescent
from unconstrained_optimization.descent_methods import SteepestDescent, STEEPEST_DESCENT_NORM_EUCLIDEAN, \
    STEEPEST_DESCENT_NORM_QUADRATIC, STEEPEST_DESCENT_NORM_L1
from unconstrained_optimization.descent_methods import NewtonStep


def select_backtracking_hyper_parameters(f, x_start, make_optimizer):
    alphas = np.linspace(0.1, 0.5, 50)
    betas = np.linspace(0.1, 0.9, 90)

    ab_pairs = list()
    ab_iterations = list()
    for alpha in alphas:
        for beta in betas:
            line_search = BacktrackingLineSearch(f=f, alpha=alpha, beta=beta)
            optimizer = make_optimizer(line_search)
            iterations, _, _, _ = optimizer.optimize(x_start)

            print("alpha=%f beta=%f iterations=%d" % (alpha, beta, iterations))
            ab_pairs.append((alpha, beta))
            ab_iterations.append(iterations)

    ab_iterations = np.array(ab_iterations)
    i = np.argmin(ab_iterations)
    alpha, beta = ab_pairs[i]

    print("Best Hyper Parameters: alpha=%f beta=%f iterations=%d" % (alpha, beta, ab_iterations[i]))
    exit(0)

def main():
    f = Function()
    x_start = np.array([1, 1])

    # Gradient descent
    # select_backtracking_hyper_parameters(
    #     f=f,
    #     x_start=x_start,
    #     make_optimizer=lambda ls: GradientDescent(f=f, line_search=ls, epsilon=1e-7)
    # )

    # Steepest Descent - Euclidean
    # select_backtracking_hyper_parameters(
    #     f=f,
    #     x_start=x_start,
    #     make_optimizer=lambda ls: SteepestDescent(
    #         f=f,
    #         line_search=ls,
    #         p=None,
    #         norm=STEEPEST_DESCENT_NORM_EUCLIDEAN,
    #         epsilon=1e-7
    #     )
    # )

    # Steepest Descent - Quadratic - P1
    # select_backtracking_hyper_parameters(
    #     f=f,
    #     x_start=x_start,
    #     make_optimizer=lambda ls: SteepestDescent(
    #         f=f,
    #         line_search=ls,
    #         p=np.array([[5, 0], [0, 5]]),
    #         norm=STEEPEST_DESCENT_NORM_QUADRATIC,
    #         epsilon=1e-7
    #     )
    # )

    # Steepest Descent - Quadratic - P2
    # select_backtracking_hyper_parameters(
    #     f=f,
    #     x_start=x_start,
    #     make_optimizer=lambda ls: SteepestDescent(
    #         f=f,
    #         line_search=ls,
    #         p=np.array([[5, 0], [0, 10]]),
    #         norm=STEEPEST_DESCENT_NORM_QUADRATIC,
    #         epsilon=1e-7
    #     )
    # )

    # Steepest Descent - Quadratic - P3
    # select_backtracking_hyper_parameters(
    #     f=f,
    #     x_start=x_start,
    #     make_optimizer=lambda ls: SteepestDescent(
    #         f=f,
    #         line_search=ls,
    #         p=np.array([[5, 0], [0, 100]]),
    #         norm=STEEPEST_DESCENT_NORM_QUADRATIC,
    #         epsilon=1e-7
    #     )
    # )

    # Steepest Descent - L1
    # select_backtracking_hyper_parameters(
    #     f=f,
    #     x_start=x_start,
    #     make_optimizer=lambda ls: SteepestDescent(
    #         f=f,
    #         line_search=ls,
    #         p=None,
    #         norm=STEEPEST_DESCENT_NORM_L1,
    #         epsilon=1e-7
    #     )
    # )

    # Newton Step
    select_backtracking_hyper_parameters(
        f=f,
        x_start=x_start,
        make_optimizer=lambda ls: NewtonStep(f=f, alpha=ls.alpha, beta=ls.beta, epsilon=1e-20)
    )


main()
