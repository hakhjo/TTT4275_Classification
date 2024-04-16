import numpy as np
from tqdm import tqdm

class NN:
    def __init__(self, template_x, template_y, C, D):
        self.template_x = template_x
        self.template_y = template_y
        self.C = C
        self.D = D
        self.M = len(template_x)

    def evaluate(self, x):
        dist = np.sum(np.square(x-self.template_x), axis=1)
        idx = np.argmin(dist)
        return self.template_y[idx]

    def confusion(self, x, y):
        errors = 0
        conf = np.zeros((self.C, self.C))
        for xk, yk in zip(tqdm(x), y):
            guess = self.evaluate(xk)
            conf[yk[0], guess[0]] += 1
            if guess[0] != yk[0]:
                errors += 1
        return conf, errors / len(x)

    def validate(self, x, y):
        errors = 0
        for xk, yk in zip(x, y):
            guess = self.evaluate(xk)
            if guess[0] != yk[0]:
                errors += 1
        return errors / len(x)