import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from classifier import classifier 
from data_modification import *
from utils import *
from matplotlib import pyplot as plt
import seaborn as sn

encoder = OneHotEncoder(sparse_output=False, categories="auto")
class_name = ["Iris Setosa","Iris Versicolour", "Iris Virginica"]
class_labels = np.array([[1], [2], [3]])
encoder.fit(class_labels)
np.random.seed(0)

feature_names = ["sepal length","sepal width","petal length","petal width"]
file_paths = ["class_1", "class_2", "class_3"]


def drop_feature(data, feature_name):
    feature_index = feature_names.index(feature_name)
    
    if isinstance(feature_index, int) and feature_index >= 0 and feature_index < data.shape[1]:
        return data.drop(data.columns[feature_index], axis=1)
    else:
        raise ValueError("Feature index is out of bounds.")


def load_and_process_data(file_paths, feature_to_drop = None):
    training_data = []
    validation_data = []
    t_labels = []
    v_labels = []
    total_data = []
    d_labels = []

    for i, file_path in enumerate(file_paths, start=1):
        data = pd.read_csv(file_path, header=None)
        if feature_to_drop != None:
            # data = drop_feature(data, "petal width")
            # data = drop_feature(data, "sepal width")
            data = drop_feature(data, feature_to_drop)
            data = drop_feature(data, "sepal length")
            
        train = data.head(30)
        validate = data.tail(20)

        training_data.append(train)
        validation_data.append(validate)
        total_data.append(data)

        label_vector = encoder.transform(np.array([[i]]))

        t_labels.append(np.repeat(label_vector, 30, axis=0))
        v_labels.append(np.repeat(label_vector, 20, axis=0))
        d_labels.append(np.repeat(label_vector, 50, axis=0))
    
    training_data = (pd.concat(training_data, ignore_index=True)).to_numpy()
    validation_data = (pd.concat(validation_data, ignore_index=True)).to_numpy()
    data = (pd.concat(total_data, ignore_index=True)).to_numpy()
    training_labels = np.vstack(t_labels)
    validation_labels = np.vstack(v_labels)
    data_labels = np.vstack(d_labels)

    return training_data, training_labels, validation_data, validation_labels, data, data_labels


def train_on_dataset(c: classifier, x, t, N, step_size, x_test, t_test):
    train_err = np.zeros(N)
    test_err = np.zeros(N)
    assert len(x) == len(t)
    for i in range(N):
        p = np.random.permutation(len(x))
        err = c.train(x[p, :], t[p, :], step_size)
        train_err[i] = err
        test_err[i] = c.validate(x_test, t_test)
        print(f"TRAINING... {i}/{N}: \t{100*err:.2f}", end="\r", flush=True)

    print("TRAINING... DONE                      ")
    return train_err, test_err



train_x, train_t, test_x, test_t, all_data, all_data_t = load_and_process_data(file_paths, "sepal width")
N = 1

# train_x, train_t, test_x, test_t, all_data, all_data_t = load_and_process_data(file_paths)
# N = 4
# train_x = reduce_dataset(N, train_x)
# test_x  = reduce_dataset(N, test_x)
c = classifier(3, N)
train_err, test_err = train_on_dataset(c, train_x, train_t, 2000, 0.001, test_x, test_t)
test_conf = c.confusion(test_x, test_t)
train_conf = c.confusion(train_x, train_t)
# plot_confusion_matrix("test confusion matrix", test_conf)
# produce_histograms(all_data, all_data_t)
# plot_confusion_matrix("training confusion matrix", train_conf)
# plot_3d_decision_boundary_between_two_classes(c,train_x, train_t)
# plot_3d_decision_boundary_between_two_classes(c,test_x , test_t)
# plot_correlation_matrix(all_data)
display_results(train_err, test_err, train_conf, test_conf)

def load_train_and_print_error(file_paths):
    train_x, train_t, test_x, test_t, all_data, all_t = load_and_process_data(file_paths)
    # Train with all features
    c = classifier(3, len(train_x[0]))
    train_err, test_err = train_on_dataset(c, train_x, train_t, 2000, 0.001, test_x, test_t)
    print(f"ERR_T with all features: {100*test_err[-1]:.2f}")
    print(f"ERR_D with all features: {100*train_err[-1]:.2f}")

    # Train with each feature removed
    for feature_name in feature_names:
        np.random.seed(0)
        train_x, train_t, test_x, test_t, all_data, all_t = load_and_process_data(file_paths, feature_name)
        c = classifier(3, len(train_x[0]))
        train_err, test_err = train_on_dataset(c, train_x, train_t, 2000, 0.001, test_x, test_t)
        print(f"EER_T without {feature_name} feature:  {100*test_err[-1]:.2f}")
        print(f"EER_D rate without {feature_name} feature:  {100*train_err[-1]:.2f}")


# load_train_and_print_error(file_paths)
# plot_correlation_matrix(all_data)
def plot_step_size_convergence():
    step_sizes = [1.0,0.1, 0.01,0.001]   
    colors = ['b', 'g', 'r', 'c']
    for i, step_size in enumerate(step_sizes):
        c = classifier(3, N)
        train_err, val_err = train_on_dataset(c, train_x, train_t, 2000, step_size, train_x, train_t)
        plt.plot(val_err, color=colors[i], alpha=0.2)
        window_size = 10
        running_avg = np.convolve(val_err, np.ones(window_size)/window_size, mode='valid')
        plt.plot(running_avg, label=f"$Step$ $Size:$ ${step_size}$",color=colors[i])
    plt.ylabel("$Error$ $rate$")
    plt.xlabel("$Iterations$")
    plt.legend()
    plt.savefig("convergence.pdf", format="pdf")

# plot_step_size_convergence()
