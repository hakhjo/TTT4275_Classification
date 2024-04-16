import numpy as np

class NN:
    def __init__(self, template_x, template_y):
        self.template_x = template_x
        self.template_y = template_y
        self.M = len(template_x)

    def evaluate(self, x):
        dist = np.sum(np.square(x-self.template_x), axis=1)
        idx = np.argmin(dist)
        return self.template_y[idx]

    def confusion(self, x, y):
        conf = np.zeros((self.C, self.C))
        for xk, yk in zip(x, y):
            guess = self.evaluate(xk)
            conf[yk, guess] += 1
        return conf

    def validate(self, x, y):
        errors = 0
        for xk, yk in zip(x, y):
            guess = self.evaluate(xk)
            if guess != yk:
                errors += 1
        return errors / len(x)