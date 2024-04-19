import numpy as np
import data_loader
from matplotlib import pyplot as plt
from classifier import NN

def display_image(vec):
    plt.imshow(vec.reshape((28, 28)))
    plt.show()

train_x, train_y, test_x, test_y = data_loader.load_data()

c = NN(train_x, train_y, 10, 784)
p = np.random.permutation(len(test_x))[:1000]
c.evaluate2(test_x[p])
exit()
test_conf, err = c.confusion(test_x, test_y)
test_conf = 100.0 * test_conf / np.sum(test_conf, axis=1)
print(f"Error rate: {err*100:.2f}")
for tr in test_conf:
    print(" ".join(f"{c:>6.2f}" for c in tr))