import pandas as pd
import numpy as np

class_names = ["Iris Setosa","Iris Versicolour", "Iris Virginica"]
feature_names = ["sepal length","sepal width","petal length","petal width"]
file_paths = ["class_1", "class_2", "class_3"]
n_features = 4
n_classes = 3

def onehot_encode(i, N, rep=1):
    if rep > 1:
        y = np.zeros((rep, N))
        y[:, i] = 1
        return y
    else:
        y = np.zeros(N)
        y[i] = 1
        return y

def drop_feature(data, feature):
    assert (feature in feature_names)
    feature_index = feature_names.index(feature)
    return np.delete(data, feature_index, axis=1)

def new_names(features):
    return [n for n in feature_names if n not in features]

def load_data(file_paths, var=1, drop_features=()):
    train_x = []
    test_x = []
    train_y = []
    test_y = []

    for i, file_path in enumerate(file_paths):
        data = pd.read_csv(file_path, header=None).to_numpy()
        for f in drop_features:
            data = drop_feature(data, f)
        
        if var == 1:
            train = data[:30]
            test = data[30:]
        elif var == 2:
            train = data[20:]
            test = data[:20]

        train_x.append(train)
        test_x.append(test)
        train_y.append(onehot_encode(i, n_classes, rep=30))
        test_y.append(onehot_encode(i, n_classes, rep=20))
    
    train_x = np.vstack(train_x)
    train_y = np.vstack(train_y)
    test_x = np.vstack(test_x)
    test_y = np.vstack(test_y)
    x = np.vstack((train_x, test_x))
    y = np.vstack((train_y, test_y))

    return train_x, train_y, test_x, test_y, x, y


if __name__ == "__main__":
    load_data(file_paths, drop_features=("sepal length",))
    print(new_names(("sepal length", "sepal width")))