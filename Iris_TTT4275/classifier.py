import numpy as np

class classifier:
    def __init__(self, C, D):
        mu = 0.
        sigma = 0.1
        self.W = np.random.normal(mu, sigma, (C, D+1))
        self.C = C
        self.D = D

    def gradient(self, x, g, t):
        return np.outer((g-t)*g*(1-g), x.T)

    def evaluate(self, x):
        return sigmoid(self.W @ x)
    
    def train(self, x, t, step):
        hits = 0
        for xk_, tk in zip(x, t):
            xk = np.append(xk_, 1)
            g = self.evaluate(xk)
            self.W = self.W - step * self.gradient(xk, g, tk)
            if np.argmax(g) == np.argmax(tk):
                hits += 1
        return hits / len(x)

    def confusion_matrix(self, x, t):
        conf = np.zeros((self.C, self.C))
        for xk_, tk in zip(x, t):
            xk = np.append(xk_, 1)
            g = self.evaluate(xk)
            conf[np.argmax(t), np.argmax(g)] += 1
        return conf


    def validate(self, x, t):
        hits = 0
        conf = np.zeros((self.C, self.C))
        for xk_, tk in zip(x, t):
            xk = np.append(xk_, 1)
            g = self.evaluate(xk)
            truth = np.argmax(tk)
            guess = np.argmax(g)
            conf[truth, guess] += 1
            if guess == truth:
                hits += 1
        return hits / len(x), conf
        

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

