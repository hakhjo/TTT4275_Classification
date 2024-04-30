import pickle
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sn

def plot_confusion_matrix(conf_mat, classes=list(i for i in "0123456789"), file=None):
    g = sn.heatmap(
        conf_mat,
        annot=True,
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
        fmt=".1%",
        annot_kws={"size": 12},
        square=True,
    )
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize=12)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize=12)
    plt.xlabel("$\\hat{\\omega}$", fontsize=18)
    plt.ylabel("$\\omega$", fontsize=18)
    if file is not None:
        plt.savefig(file, bbox_inches="tight")
        plt.clf()
    else:
        plt.show()


def display_image(vec):
    plt.imshow(vec.reshape((28, 28)))
    plt.show()


def confusion_matrix(predictions, truths, N, relative=False):
    assert predictions.shape == truths.shape
    mat = np.zeros((N, N), dtype=float if relative else int)
    print(predictions.shape)
    for p, t in zip(predictions, truths):
        mat[t, p] += 1
    if relative:
        return mat / np.sum(mat, axis=1)
    else:
        return mat


def confusion_percentage(mat):
    return mat / np.sum(mat, axis=1)


def error_rate(predictions, truths):
    assert predictions.shape == truths.shape
    errors = 0
    for p, t in zip(predictions, truths):
        errors += p != t
    return errors / len(predictions)


def mat_percentage_tostring(mat):
    return "\n".join((" ".join(f"{100.*elem:>6.2f}" for elem in row) for row in mat))


def mat_tostring(mat):
    h2 = "  |" + " ".join(f"{elem:>4}" for elem in range(mat.shape[1]))
    h11 = "⬐ TRUTH"
    h1 = h11 + "GUESS".center(len(h2))[len(h11) :]
    h3 = "--+" + "-" * (len(h2) - 3)
    s = "\n".join(
        (f"{i} |" + " ".join(f"{elem:>4}" for elem in row) for i, row in enumerate(mat))
    )
    return f"{h1}\n{h2}\n{h3}\n{s}"


def wilson_CI(p0, n, z_a=1.96):
    q0 = 1 - p0
    R = z_a * np.sqrt(p0 * q0 / n + z_a**2 / (4 * n**2)) / (1 + (z_a**2) / n)
    p = (p0 + (z_a**2) / (2 * n)) / (1 + z_a**2 / n)
    return p - R, p + R


def plot_centroids(cluster_centers, M, N):
    centroids = cluster_centers.reshape((N * M, 28, 28))
    for i in range(N):
        for j in range(M):
            plt.subplot(N, M, i * M + j + 1)
            plt.imshow(centroids[i * M + j], cmap="gray")
            plt.axis("off")
    plt.show()


class Result:
    def __init__(self, pred, truth, t, n_classes, title):
        self.pred = pred
        self.truth = truth
        self.time = t
        self.conf = confusion_matrix(pred, truth, n_classes)
        self.err = error_rate(pred, truth)
        self.title = title

    def report(self):
        print(
            f"""
REPORT: {self.title}
--------{"-"*len(self.title)}
 - Prediction time: {self.time:.1f}s
 - Error rate:      {100*self.err:.2f}%

CONFUSION MATRIX:
{mat_tostring(self.conf)}

"""
        )

    def save(this, fname):
        with open(fname, "wb") as f:
            return pickle.dump(this, f)


def result_from_file(fname) -> Result:
    with open(fname, "rb") as f:
        return pickle.load(f)
