import numpy as np
import data_loader
from matplotlib import pyplot as plt
from classifier import NN

def display_image(vec):
    plt.imshow(vec.reshape((28, 28)))
    plt.show()

train_x, train_y, test_x, test_y = data_loader.load_data()
c = NN(train_x, train_y)
print(c.evaluate(test_x[0]), test_y[0])
display_image(test_x[0])