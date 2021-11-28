import numpy as np


class Function:
    def __call__(self, x):
        return np.sum(x * np.log(x))
