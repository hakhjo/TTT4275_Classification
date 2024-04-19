import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from classifier import classifier 
from data_modification import *
from matplotlib import pyplot as plt
import seaborn as sn

encoder = OneHotEncoder(sparse_output=False, categories="auto")

class_labels = np.array([[1], [2], [3]])
encoder.fit(class_labels)


def load_and_process_data(file_paths):
    training_data = []
    validation_data = []
    t_labels = []
    v_labels = []

    for i, file_path in enumerate(file_paths, start=1):
        data = pd.read_csv(file_path, header=None)
        train = data.head(30)
        validate = data.tail(20)

        training_data.append(train)
        validation_data.append(validate)

        label_vector = encoder.transform(np.array([[i]]))
        t_labels.append(np.repeat(label_vector, 30, axis=0))
        v_labels.append(np.repeat(label_vector, 20, axis=0))
    training_data = (pd.concat(training_data, ignore_index=True)).to_numpy()
    validation_data = (pd.concat(validation_data, ignore_index=True)).to_numpy()
    training_labels = np.vstack(t_labels)
    validation_labels = np.vstack(v_labels)
    return training_data, training_labels, validation_data, validation_labels


def train_on_dataset(c: classifier, x, t, N, step_size, x_val, t_val):
    train_err = np.zeros(N)
    val_err = np.zeros(N)
    assert len(x) == len(t)
    for i in range(N):
        p = np.random.permutation(len(x))
        err = c.train(x[p, :], t[p, :], step_size)
        train_err[i] = err
        val_err[i] = c.validate(x_val, t_val)
        print(f"TRAINING... {i}/{N}: \t{100*err:.2f}", end="\r", flush=True)

    print("TRAINING... DONE                      ")
    return train_err, val_err


def display_results(train_err, val_err, train_conf, val_conf):
    val_conf = 100.0 * val_conf / np.sum(val_conf, axis=1)
    train_conf = 100.0 * train_conf / np.sum(train_conf, axis=1)
    print("\n------------- RESULTS ---------------")
    print("    Validation:            Training:")
    print(f"ERROR RATE: {100 * val_err[-1]:.2f}            {100 * train_err[-1]:.2f}")
    print("CONFUSION MATRICES")
    for vr, tr in zip(val_conf, train_conf):
        print(
            " ".join(f"{c:>6.2f}" for c in vr),
            " | ",
            " ".join(f"{c:>6.2f}" for c in tr),
        )

    plt.plot(train_err, "r", label="training error")
    plt.plot(val_err, "b", label="validation error")
    plt.xlabel("Iteration")
    plt.ylabel("Error rate")
    plt.legend()
    plt.show()


file_paths = ["class_1", "class_2", "class_3"]

train_x, train_t, val_x, val_t = load_and_process_data(
    file_paths
)

produce_histograms(train_x, train_t)
N = 2
train_x = reduce_dataset(N, train_x, True)
val_x = reduce_dataset(N, val_x)
c = classifier(3, N)
train_err, val_err = train_on_dataset(
    c, train_x, train_t, 2000, 0.001,train_x, train_t
)
val_conf = c.confusion(val_x, val_t)
train_conf = c.confusion(train_x, train_t)
plot_confusion_matrix("confusion matrix",val_conf)

display_results(train_err, val_err, train_conf, val_conf)
# reduce_dataset(3,train_features)
# print("Validation Features:", val_features)
# print(training_set)
