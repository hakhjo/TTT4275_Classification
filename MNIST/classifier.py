import numpy as np
from tqdm import tqdm, trange


class NNClassifier:
    def __init__(self, template_x, template_y, chunk_size=None):
        self.n_templates = len(template_x)

        self.template_x = template_x
        self.template_y = template_y

        if chunk_size is None:
            self.chunk_size = self.n_templates
        else:
            self.chunk_size = chunk_size
        self.n_chunks = int(np.ceil(self.n_templates / self.chunk_size))
        self.template_x_chunked = list(
            [
                self.template_x[
                    i * self.chunk_size : min(self.n_templates, ((i + 1) * self.chunk_size))
                ]
                for i in range(self.n_chunks)
            ]
        )
        self.template_y_chunked = list(
            [
                self.template_y[
                    i * self.chunk_size : min(self.n_templates, ((i + 1) * self.chunk_size))
                ]
                for i in range(self.n_chunks)
            ]
        )

    def predict_single(self, x):
        dist = np.sum(np.square(x - self.template_x), axis=1)
        idx = np.argmin(dist)
        return (self.template_y)[idx]

    def predict_chunk(self, x):
        minima = np.zeros((len(x), self.n_chunks))
        minima_labels = np.zeros_like(minima, dtype=int)
        for i in trange(self.n_chunks):
            diff = x[:, np.newaxis, :] - self.template_x_chunked[i][np.newaxis, :, :]
            dist = np.sum(np.square(diff), axis=2)
            min_idx = np.argmin(dist, axis=1)
            minima[:, i] = dist[range(len(x)), min_idx]
            minima_labels[:, i] = self.template_y_chunked[i][min_idx, 0]
        glb_min_idx = np.argmin(minima, axis=1)
        predictions = minima_labels[range(len(x)), glb_min_idx]
        return predictions

    def predict_array(self, x, chunk_size=None):
        N = len(x)
        if chunk_size is None:
            chunk_size = N
        n_chunks = int(np.ceil(N / chunk_size))
        chunked_x = list(
            [
                x[i * chunk_size : min(N, ((i + 1) * chunk_size))]
                for i in range(n_chunks)
            ]
        )
        predictions = np.hstack([self.predict_chunk(chunk) for chunk in tqdm(chunked_x)])
        return predictions


# class KNNClassifier:
#     def __init__(self, template_x, template_y, n_classes, chunk_size=None):
#         self.n_classes = n_classes
#         self.n_templates = len(template_x)

#         self.template_x = template_x
#         self.template_y = template_y

#         if chunk_size is None:
#             self.chunk_size = n_classes
#         else:
#             self.chunk_size = chunk_size
#         self.n_chunks = int(np.ceil(self.n_templates / self.chunk_size))
#         self.template_x_chunked = list(
#             [
#                 self.template_x[
#                     i * self.chunk_size : min(self.n_templates, ((i + 1) * self.chunk_size))
#                 ]
#                 for i in range(self.n_chunks)
#             ]
#         )
#         self.template_y_chunked = list(
#             [
#                 self.template_y[
#                     i * self.chunk_size : min(self.n_templates, ((i + 1) * self.chunk_size))
#                 ]
#                 for i in range(self.n_chunks)
#             ]
#         )

#     def predict_single(self, x):
#         dist = np.sum(np.square(x - self.template_x), axis=1)
#         idx = np.argmin(dist)
#         return (self.template_y)[idx]

#     def predict_chunk(self, x):
#         minima = np.zeros((len(x), self.n_chunks))
#         minima_labels = np.zeros_like(minima, dtype=int)
#         for i in trange(self.n_chunks):
#             diff = x[np.newaxis, :, :] - self.template_x_chunked[i][:, np.newaxis, :]
#             dist = np.sum(np.square(diff), axis=2)
#             min_idx = np.argmin(dist, axis=1)
#             minima[:, i] = dist[range(len(x)), min_idx]
#             minima_labels[:, i] = self.template_y_chunked[i][min_idx, 0]
#         glb_min_idx = np.argmin(minima, axis=1)
#         predictions = minima_labels[range(len(x)), glb_min_idx]
#         return predictions

#     def evaluate_array(self, x, chunk_size=None):
#         N = len(x)
#         if chunk_size is None:
#             chunk_size = N
#         n_chunks = int(np.ceil(N / chunk_size))
#         chunked_x = list(
#             [
#                 x[i * chunk_size : min(N, ((i + 1) * chunk_size))]
#                 for i in range(n_chunks)
#             ]
#         )
#         predictions = np.hstack([self.predict_chunk(chunk) for chunk in tqdm(chunked_x)])
#         return predictions


#     def confusion(self, x, y):
#         errors = 0
#         samples = 0
#         conf = np.zeros((self.n_classes, self.n_classes))
#         for i in trange(len(x)):
#             samples += len(x[i])
#             pred = self.predict_chunk(x[i])
#             for guess, yk in zip(pred, y[i]):
#                 conf[yk[0], guess] += 1
#                 if guess != yk[0]:
#                     errors += 1
#         return conf, errors / samples

#     def validate(self, x, y):
#         errors = 0
#         for xk, yk in zip(x, y):
#             guess = self.evaluate(xk)
#             if guess[0] != yk[0]:
#                 errors += 1
#         return errors / len(x)