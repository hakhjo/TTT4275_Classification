import numpy as np
import utils
from matplotlib import pyplot as plt
from classifier import NNClassifier, ClusteredKNNClassifier
import time

np.random.seed(0)

train_x, train_y, test_x, test_y = utils.load_data()

M = 64
N = 10

def run_KNN_timed(c: ClusteredKNNClassifier, K):
    start = time.time()
    test_predictions = c.predict_array(test_x, K)
    test_time = time.time() - start
    res = utils.Result(test_predictions, test_y[:, 0], test_time, N, f"{K}-NN, entire test set")
    res.report()
    res.save(f"{K}-NN_cmplt")
    return res

def run_NN_timed(c: NNClassifier):
    start = time.time()
    test_predictions = c.predict_array(test_x, chunk_size=1000)
    test_time = time.time() - start
    res = utils.Result(test_predictions, test_y[:, 0], test_time, N, "NN, entire test set")
    res.report()
    res.save("NN_cmplt")
    return res






c = NNClassifier(train_x, train_y, chunk_size=1000)
run_NN_timed(c)

start_cluster = time.time()
c = ClusteredKNNClassifier(train_x, train_y, N, M)
cluster_time = time.time() - start_cluster

print(f"Clustering took {cluster_time}s")

run_KNN_timed(c, 1)
run_KNN_timed(c, 7)

