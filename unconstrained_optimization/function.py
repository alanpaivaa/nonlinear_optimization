import numpy as np


class Function:
    def __call__(self, x):
        return self.e_x1_3x2_01(x) + self.e_x1_2x2_01(x) + self.e_x1_02(x)

    def gradient(self, x):
        return np.array([self.der_x1(x), self.der_x2(x)])

    def hessian(self, x):
        return np.array([
            [self.der2_x1_x1(x), self.der2_x1_x2(x)],
            [self.der2_x2_x1(x), self.der2_x2_x2(x)]
        ])

    @staticmethod
    def e_x1_3x2_01(x):
        return np.exp(x[0] + 3 * x[1] - 0.1)

    @staticmethod
    def e_x1_2x2_01(x):
        return np.exp(x[0] - 2 * x[1] - 0.1)

    @staticmethod
    def e_x1_02(x):
        return np.exp(-x[0] - 0.2)

    def der_x1(self, x):
        return self.e_x1_3x2_01(x) + self.e_x1_2x2_01(x) - self.e_x1_02(x)

    def der_x2(self, x):
        return 3 * self.e_x1_3x2_01(x) - 2 * self.e_x1_2x2_01(x)

    def der2_x1_x1(self, x):
        return self.e_x1_3x2_01(x) + self.e_x1_2x2_01(x) + self.e_x1_02(x)

    def der2_x1_x2(self, x):
        return self.der_x2(x)

    def der2_x2_x1(self, x):
        return self.der_x2(x)

    def der2_x2_x2(self, x):
        return 9 * self.e_x1_3x2_01(x) + 4 * self.e_x1_2x2_01(x)
