import numpy as np
import scipy.io

def load_data():
    f = scipy.io.loadmat("data_all.mat")
    testv = np.array(f["testv"])
    trainv = np.array(f["trainv"])
    testlab = np.array(f["testlab"])
    trainlab = np.array(f["trainlab"])
    return trainv, trainlab, testv, testlab