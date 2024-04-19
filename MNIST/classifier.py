import numpy as np
from tqdm import trange


class NNClassifier:
    def __init__(self, template_x, template_y, C, D, chunk_size=1000):
        self.chunk_size = chunk_size
        self.n_chunks = int(np.ceil(len(template_x) / chunk_size))
        self.template_x = template_x
        self.template_y = template_y
        self.template_x_chunked = list(
            [
                template_x[
                    i * chunk_size : min(len(template_x), ((i + 1) * chunk_size))
                ]
                for i in range(self.n_chunks)
            ]
        )
        self.template_y_chunked = list(
            [
                template_y[
                    i * chunk_size : min(len(template_y), ((i + 1) * chunk_size))
                ]
                for i in range(self.n_chunks)
            ]
        )
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
        for i in trange(self.n_chunks):
            diff = x[np.newaxis, :, :] - self.template_x_chunked[i][:, np.newaxis, :]
            dist = np.sum(np.square(diff), axis=2)
            min_idx = np.argmin(dist, axis=1)
            minima[:, i] = dist[range(len(x)), min_idx]
            minima_labels[:, i] = self.template_y_chunked[i][min_idx, 0]
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
