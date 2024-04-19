import numpy as np
from tqdm import tqdm

class NN:
    def __init__(self, template_x, template_y, C, D):
        self.template_x = template_x
        self.template_y = template_y
        self.C = C
        self.D = D
        self.M = len(template_x)

    def evaluate_single(self, x):
        dist = np.sum(np.square(x - self.template_x), axis=1)
        idx = np.argmin(dist)
        return (self.template_y)[idx]

    def evaluate_array(self, x):
        minima = np.zeros((len(x), self.n_chunks))
        minima_labels = np.zeros_like(minima, dtype=int)
        for i in tqdm(range(self.n_chunks)):
            x_tiled = np.transpose(np.tile(x, (self.chunk_size, 1, 1)), (1, 0, 2))
            temp_tiled = np.tile(self.template_x[i], (x.shape[0], 1, 1))
            dist = np.sum(np.square(x_tiled-temp_tiled), axis=2)
            min_idx = np.argmin(dist, axis=1)
            minima[:, i] = dist[range(len(x)), min_idx]
            minima_labels[:, i] = self.template_y[i][min_idx, 0]
            # print(dist[range(len(x)), min_idx].shape)
        glb_min_idx = np.argmin(minima, axis=1)
        predictions = minima_labels[range(len(x)), glb_min_idx]
        return predictions

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