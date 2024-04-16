import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from classifier import classifier 
from data_modification import reduce_dataset, produce_histograms

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


def train_on_dataset(c: classifier, x, t, N):
    assert len(x) == len(t)
    print("TRAINING...")
    for i in range(N):
        p = np.random.permutation(len(x))
        acc = c.train(x[p, :], t[p, :], 0.001)
        print(f"{i}/{N}: \t{100*acc:.2f}", end="\r", flush=True)

    print("DONE                                        ")


file_paths = ["class_1", "class_2", "class_3"]
# Load the data
train_features, train_labels, val_features, val_labels = load_and_process_data(file_paths)
N = 4
produce_histograms(train_features, train_labels)

reduced_training = reduce_dataset(N,train_features)
reduced_validate = reduce_dataset(N, val_features)
c = classifier(3, N)
train_on_dataset(c, reduced_training, train_labels, 100)
val_acc, val_conf = c.validate(reduced_validate, val_labels)
train_acc, train_conf = c.validate(reduced_training, train_labels)
val_conf_percent = 100.0 * val_conf / np.sum(val_conf, axis=1)
train_conf_percent = 100.0 * train_conf / np.sum(train_conf, axis=1)
print(f"Validation accuray: {100 * val_acc:.2f}")
print(f"Training accuray: {100 * train_acc:.2f}")
print("CONFUSION MATRICES\nValidation:            Training:")
for vr, tr in zip(val_conf_percent, train_conf_percent):
    print(" ".join(f"{c:>6.2f}" for c in vr), "|", " ".join(f"{c:>6.2f}" for c in tr))


# print("Validation Features:", val_features)
# print(training_set)
