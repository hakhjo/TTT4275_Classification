import numpy as np
import sklearn.neighbors
from tqdm import tqdm, trange
import sklearn.cluster
import collections
from time import sleep

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
                    i
                    * self.chunk_size : min(
                        self.n_templates, ((i + 1) * self.chunk_size)
                    )
                ]
                for i in range(self.n_chunks)
            ]
        )
        self.template_y_chunked = list(
            [
                self.template_y[
                    i
                    * self.chunk_size : min(
                        self.n_templates, ((i + 1) * self.chunk_size)
                    )
                ]
                for i in range(self.n_chunks)
            ]
        )

    def get_nn(self, x):
        dist = np.sum(np.square(x - self.template_x), axis=1)
        idx = np.argmin(dist)
        return self.template_y[idx], self.template_x[idx]

    def predict_single(self, x):
        dist = np.sum(np.square(x - self.template_x), axis=1)
        idx = np.argmin(dist)
        return self.template_y[idx]

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
        predictions = np.hstack(
            [self.predict_chunk(chunk) for chunk in tqdm(chunked_x)]
        )
        return predictions


class ClusteredKNNClassifier:
    def __init__(self, template_x, template_y, n_classes, clusters_per_class):
        self.n_classes = n_classes
        self.n_templates = len(template_x)

        self.template_x, self.template_y = ClusteredKNNClassifier.cluster_templates(
            template_x,
            template_y,
            clusters_per_class,
            n_classes,
        )

    def cluster_templates(x, y, clusters_per_class, n_classes):
        cluster_centers = np.zeros((n_classes * clusters_per_class, x.shape[1]))
        cluster_labels = np.zeros((n_classes * clusters_per_class,), dtype=int)
        for i in range(n_classes):
            indices = np.where(y == i)[0]
            kmeans = sklearn.cluster.KMeans(n_clusters=clusters_per_class)
            kmeans.fit(x[indices])
            cluster_centers[i * clusters_per_class : (i + 1) * clusters_per_class] = (
                kmeans.cluster_centers_
            )
            cluster_labels[i * clusters_per_class : (i + 1) * clusters_per_class] = i
        return cluster_centers, cluster_labels

    def predict_single(self, x, K):
        dist = np.sum(np.square(x - self.template_x), axis=1)
        lowest_indices = np.argpartition(dist, K)[:K]
        lowest_dist = dist[lowest_indices]
        sort_indices = np.argsort(lowest_dist)
        lowest_dist = lowest_dist[sort_indices]
        lowest_labels = self.template_y[lowest_indices][sort_indices]
        pred = ordered_majority_vote(lowest_labels)
        return pred

    def predict_array(self, x, K):
        predictions = [self.predict_single(xk, K) for xk in tqdm(x)]
        return np.hstack(predictions)
    
    def get_nn(self, x):
        dist = np.sum(np.square(x - self.template_x), axis=1)
        idx = np.argmin(dist)
        return self.template_y[idx], self.template_x[idx]


def ordered_majority_vote(x):
    c = collections.OrderedDict()
    for i in x:
        if i in c:
            c[i] += 1
        else:
            c[i] = 1
    maxidx = np.argmax(list(c.values()))
    return list(c.keys())[maxidx]
