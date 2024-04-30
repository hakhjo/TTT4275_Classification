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
            # self.W = self.W - step * self.gradient(xk, g, tk) Commented out for Batch approch
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

    def train_on_dataset(self, train_x, train_y, N, step_size, test_x, test_y):
        train_err = np.zeros(N)
        test_err = np.zeros(N)
        assert len(train_x) == len(train_y)
        for i in range(N):
            p = np.random.permutation(len(train_x))
            err = self.train(train_x[p, :], train_y[p, :], step_size)
            train_err[i] = err
            test_err[i] = self.validate(test_x, test_y)
            print(f"TRAINING... {i}/{N}: \t{100*err:.2f}", end="\r", flush=True)

        print("TRAINING... DONE                      ")
        return train_err, test_err
        

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

