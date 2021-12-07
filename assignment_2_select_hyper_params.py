import numpy as np
from constrained_optimization.function import Function
from constrained_optimization.descent_methods import FeasibleStartNewtonStep, InfeasibleStartNewtonStep


def select_backtracking_hyper_parameters(f, x_start, make_optimizer):
    alphas = np.linspace(0.1, 0.5, 50)
    betas = np.linspace(0.1, 0.9, 90)

    ab_pairs = list()
    ab_iterations = list()
    for alpha in alphas:
        for beta in betas:
            optimizer = make_optimizer(alpha, beta)
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
    # Set the seed to have deterministic approach
    np.random.seed(11)

    # Define function
    f = Function()

    # Generate problem
    a = np.random.rand(30, 100)
    x_hat = np.random.uniform(size=100)
    b = a @ x_hat

    # Feasible starting point
    # select_backtracking_hyper_parameters(
    #     f=f,
    #     x_start=x_hat,
    #     make_optimizer=lambda alpha, beta: FeasibleStartNewtonStep(
    #         f=f,
    #         a=a,
    #         b=b,
    #         alpha=alpha,
    #         beta=beta,
    #         epsilon=1e-20
    #     )
    # )

    # Infeasible starting point
    select_backtracking_hyper_parameters(
        f=f,
        x_start=np.ones(x_hat.shape) * 100,
        make_optimizer=lambda alpha, beta: InfeasibleStartNewtonStep(
            f=f,
            a=a,
            b=b,
            alpha=alpha,
            beta=beta,
            epsilon=1e-13
        )
    )


main()
