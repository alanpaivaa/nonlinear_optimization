import time
import numpy as np
from constrained_optimization.function import Function
from constrained_optimization.descent_methods import FeasibleStartNewtonStep, InfeasibleStartNewtonStep
from constrained_optimization.plot_helper import plot_error_curve


def main():
    # Load a and x_hat from file
    a = np.load('constrained_optimization/a.npy')
    x_hat = np.load('constrained_optimization/x_hat.npy')
    b = a @ x_hat

    print("A has full rank: %s" % (np.linalg.matrix_rank(a) == a.shape[0]))

    # Define function
    f = Function()

    # Feasible starting point optimization
    optimizer = FeasibleStartNewtonStep(f=f, a=a, b=b, alpha=0.1, beta=0.1, epsilon=1e-20)
    start_time = time.time()
    iterations, x_min_feasible, _, errors = optimizer.optimize(x_hat)
    end_time = time.time()

    # Print results
    print("\n----- Feasible Starting Point -----")
    print("Número de Iterações: %d" % iterations)
    print("f(x*) = %.10f" % f(x_min_feasible))
    print("Tempo de execução: %d ms" % ((end_time - start_time) * 1000))

    # Plot error curve
    plot_error_curve(errors=errors)

    # Infeasible starting point optimization
    optimizer = InfeasibleStartNewtonStep(f=f, a=a, b=b, alpha=0.1, beta=0.594382, epsilon=1e-11)
    start_time = time.time()
    iterations, x_min_infeasible, _, errors = optimizer.optimize(np.ones(x_hat.shape) * 100)
    end_time = time.time()

    # Plot error curve
    plot_error_curve(errors=errors)
    
    # Print results
    print("\n----- Infeasible Starting Point -----")
    print("Número de Iterações: %d" % iterations)
    print("f(x*) = %.10f" % f(x_min_infeasible))
    print("Tempo de execução: %d ms" % ((end_time - start_time) * 1000))

    # Check if both minimum points are equal
    print("\nFeasible and infeasible points are equal: %s" % np.allclose(x_min_feasible, x_min_infeasible))


main()
