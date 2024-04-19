import numpy as np
import data_loader
from matplotlib import pyplot as plt
from classifier import NN


def display_image(vec):
    plt.imshow(vec.reshape((28, 28)))
    plt.show()


train_x, train_y, test_x, test_y = data_loader.load_data()

chunk_size = 1000
n_chunks = int(np.ceil(len(test_x) / chunk_size))
chunked_test_x = list(
    [
        test_x[i * chunk_size : min(len(test_x), ((i + 1) * chunk_size))]
        for i in range(n_chunks)
    ]
)
chunked_test_y = list(
    [
        test_y[i * chunk_size : min(len(test_y), ((i + 1) * chunk_size))]
        for i in range(n_chunks)
    ]
)


c = NN(train_x, train_y, 10, 784)
test_conf, err = c.confusion(chunked_test_x, chunked_test_y)

test_conf = 100.0 * test_conf / np.sum(test_conf, axis=1)
print(f"Error rate: {err*100:.2f}")
for tr in test_conf:
    print(" ".join(f"{c:>6.2f}" for c in tr))
