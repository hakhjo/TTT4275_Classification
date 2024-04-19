import numpy as np
from tqdm import tqdm, trange


class NN:
    def __init__(self, template_x, template_y, C, D, chunk_size=1000):
        self.chunk_size = chunk_size
        self.n_chunks = int(np.ceil(len(template_x) / chunk_size))
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

    def evaluate_single(self, x):
        dist = np.sum(np.square(x - np.vstack(self.template_x)), axis=1)
        idx = np.argmin(dist)
        return np.vstack(self.template_y)[idx]

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
        samples = 0
        conf = np.zeros((self.C, self.C))
        for i in trange(len(x)):
            samples += len(x[i])
            pred = self.evaluate_array(x[i])
            for guess, yk in zip(pred, y[i]):
                conf[yk[0], guess] += 1
                if guess != yk[0]:
                    errors += 1
        return conf, errors / samples

    def validate(self, x, y):
        errors = 0
        for xk, yk in zip(x, y):
            guess = self.evaluate(xk)
            if guess[0] != yk[0]:
                errors += 1
        return errors / len(x)
