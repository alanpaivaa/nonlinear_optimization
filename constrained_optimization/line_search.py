class BacktrackingLineSearch:
    def __init__(self, f, alpha, beta):
        self.f = f
        self.alpha = alpha
        self.beta = beta

    def __call__(self, x, delta_x):
        t = 1
        while self.f(x + t * delta_x) > self.f(x) + self.alpha * t * (self.f.gradient(x) @ delta_x.reshape(x.shape[0], 1)).item():
            t = self.beta * t
        return t
