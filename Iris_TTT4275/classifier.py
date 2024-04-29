import numpy as np

class classifier:
    def __init__(self, C, D):
        mu = 0.
        sigma = 0.1
        self.W = np.random.normal(mu, sigma, (C, D+1))
        self.C = C

    def gradient(self, x, g, t):
        return np.outer((g-t)*g*(1-g), x.T)

    def evaluate(self, x):
        return sigmoid(self.W @ x)
    
    def train(self, x, t, step):
        grad = np.zeros_like(self.W)
        errors = 0
        for xk_, tk in zip(x, t):
            xk = np.append(xk_, 1)
            g = self.evaluate(xk)
            grad += self.gradient(xk, g, tk)
            # self.W = self.W - step * self.gradient(xk, g, tk)
            if np.argmax(g) != np.argmax(tk):
                errors += 1
        self.W = self.W - step * grad
        return errors / len(x)

    def confusion(self, x, t):
        conf = np.zeros((self.C, self.C))
        for xk_, tk in zip(x, t):
            xk = np.append(xk_, 1)
            g = self.evaluate(xk)
            truth = np.argmax(tk)
            guess = np.argmax(g)
            conf[truth, guess] += 1
        return conf

    def validate(self, x, t):
        errors = 0
        for xk_, tk in zip(x, t):
            xk = np.append(xk_, 1)
            g = self.evaluate(xk)
            truth = np.argmax(tk)
            guess = np.argmax(g)
            if guess != truth:
                errors += 1
        return errors / len(x)
        

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

