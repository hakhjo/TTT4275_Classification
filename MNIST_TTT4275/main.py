import numpy as np
import utils
from classifier import NNClassifier, ClusteredKNNClassifier
import scipy
import time


def load_data():
    f = scipy.io.loadmat("data_all.mat")
    testv = np.array(f["testv"], dtype=np.int32)
    trainv = np.array(f["trainv"], dtype=np.int32)
    testlab = np.array(f["testlab"], dtype=np.int32)
    trainlab = np.array(f["trainlab"], dtype=np.int32)
    return trainv, trainlab, testv, testlab


def run_KNN_timed(c: ClusteredKNNClassifier, K):
    start = time.time()
    test_predictions = c.predict_array(test_x, K)
    test_time = time.time() - start
    res = utils.Result(
        test_predictions, test_y[:, 0], test_time, 10, f"{K}-NN, entire test set"
    )
    res.report()
    res.save(f"{K}-NN_cmplt")
    return res


def run_NN_timed(c: NNClassifier):
    start = time.time()
    test_predictions = c.predict_array(test_x, chunk_size=1000)
    test_time = time.time() - start
    res = utils.Result(
        test_predictions, test_y[:, 0], test_time, 10, "NN, entire test set"
    )
    res.report()
    res.save("NN_cmplt")
    return res

np.random.seed(0)
train_x, train_y, test_x, test_y = load_data()

c = NNClassifier(
    train_x,
    train_y,
    chunk_size=1000,
)
run_NN_timed(c)

c = ClusteredKNNClassifier(
    train_x,
    train_y,
    n_classes=10,
    clusters_per_class=64,
)
run_KNN_timed(c, 1)
run_KNN_timed(c, 7)
