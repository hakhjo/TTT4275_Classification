import numpy as np
import data_loader
from matplotlib import pyplot as plt
from classifier import NN

def display_image(vec):
    plt.imshow(vec.reshape((28, 28)))
    plt.show()

train_x, train_y, test_x, test_y = data_loader.load_data()
# print(train_x.shape)
p = np.random.permutation(len(train_x))[:1000]
c = NN(train_x[p], train_y[p], 10, 784)
test_conf, err = c.confusion(test_x, test_y)
# train_conf = c.confusion(train_x, train_y)
test_conf = 100.0 * test_conf / np.sum(test_conf, axis=1)
# train_conf = 100.0 * train_conf / np.sum(train_conf, axis=1)
print(f"Error rate: {err*100:.2f}")
for tr in test_conf:
    print(" ".join(f"{c:>6.2f}" for c in tr))
# print(c.evaluate(test_x[0]), test_y[0])
# display_image(test_x[0])