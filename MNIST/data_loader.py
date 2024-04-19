import numpy as np
import scipy.io

def load_data():
    f = scipy.io.loadmat("data_all.mat")
    testv = np.array(f["testv"], dtype=np.int32)
    trainv = np.array(f["trainv"], dtype=np.int32)
    testlab = np.array(f["testlab"], dtype=np.int32)
    trainlab = np.array(f["trainlab"], dtype=np.int32)
    return trainv, trainlab, testv, testlab