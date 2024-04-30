from classifier import classifier
from data import *

np.random.seed(0)
remove_features = []
train_x, train_y, test_x, test_y, x, y = load_data(1, remove_features)
c = classifier(n_classes, n_features - len(remove_features))
c.train_on_dataset(train_x, train_y, 1500, 0.005, test_x, test_y)
