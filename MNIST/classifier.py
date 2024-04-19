import numpy as np
from tqdm import tqdm


class NN:
    def __init__(self, template_x, template_y, C, D, chunk_size=1000):
        self.chunk_size = chunk_size
        self.n_chunks = int(np.floor(len(template_x) / chunk_size))
        self.template_x = list(
            [
                template_x[i * chunk_size : min(len(template_x), ((i + 1) * chunk_size))]
                for i in range(self.n_chunks)
            ]
        )
        self.template_y = list(
            [
                template_y[i * chunk_size : min(len(template_y), ((i + 1) * chunk_size))]
                for i in range(self.n_chunks)
            ]
        )
        self.C = C
        self.D = D
        self.M = len(template_x)

    def evaluate(self, x):
        dist = np.sum(np.square(x - self.template_x), axis=1)
        idx = np.argmin(dist)
        return self.template_y[idx]

    def evaluate2(self, x):
        minima = np.zeros_like(self.n_chunks)
        for i in range(self.n_chunks):
            x_tiled = np.transpose(np.tile(x, (self.chunk_size, 1, 1)), (1, 0, 2))
            temp_tiled = np.tile(self.template_x[i], (x.shape[0], 1, 1))
            dist = np.sum(np.square(x_tiled-temp_tiled), axis=2)

        return

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
