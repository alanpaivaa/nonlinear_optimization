import numpy as np


class Function:
    def __call__(self, x):
        return np.sum(x * np.log(x))

    @staticmethod
    def gradient(x):
        return 1 + np.log(x)

    @staticmethod
    def hessian(x):
        hes = np.zeros((x.shape[0], x.shape[0]))
        np.fill_diagonal(hes, 1 / x)
        return hes
