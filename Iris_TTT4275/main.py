import pandas as pd
import numpy as np
from classifier import classifier 
from data_modification import *
from utils import *
from matplotlib import pyplot as plt
import seaborn as sn
from data import *

np.random.seed(0)


def train_on_dataset(c: classifier, train_x, train_y, N, step_size, test_x, test_y):
    train_err = np.zeros(N)
    test_err = np.zeros(N)
    assert len(train_x) == len(train_y)
    for i in range(N):
        p = np.random.permutation(len(train_x))
        err = c.train(train_x[p, :], train_y[p, :], step_size)
        train_err[i] = err
        test_err[i] = c.validate(test_x, test_y)
        print(f"TRAINING... {i}/{N}: \t{100*err:.2f}", end="\r", flush=True)

    print("TRAINING... DONE                      ")
    return train_err, test_err



# train_x, train_y, test_x, test_y, x, y = load_data(file_paths, drop_features="sepal width")
# c = classifier(n_classes, n_features-1)
# train_err, test_err = train_on_dataset(c, train_x, train_y, 2000, 0.001, test_x, test_y)
# test_conf = c.confusion(test_x, test_y)
# train_conf = c.confusion(train_x, train_y)


# plot_confusion_matrix("test confusion matrix", test_conf)
# produce_histograms(all_data, all_data_t)
# plot_confusion_matrix("training confusion matrix", train_conf)
# plot_3d_decision_boundary_between_two_classes(c,train_x, train_t)
# plot_3d_decision_boundary_between_two_classes(c,test_x , test_t)
# plot_correlation_matrix(all_data)


# display_results(train_err, test_err, train_conf, test_conf)

def load_train_and_print_error(file_paths):
    train_x, train_y, test_x, test_y, x, y = load_data()
    # Train with all features
    c = classifier(3, len(train_x[0]))
    train_err, test_err = train_on_dataset(c, train_x, train_y, 1500, 0.005, test_x, test_y)
    print(f"ERR_T with all features: {100*test_err[-1]:.2f}")
    print(f"ERR_D with all features: {100*train_err[-1]:.2f}")

    # Train with each feature removed
    for feature_name in feature_names:
        np.random.seed(0)
        train_x, train_y, test_x, test_y, x, y = load_data(drop_features=feature_name)
        c = classifier(3, len(train_x[0]))
        train_err, test_err = train_on_dataset(c, train_x, train_y, 1500, 0.005, test_x, test_y)
        print(f"EER_T without {feature_name}:  {100*test_err[-1]:.2f}")
        print(f"EER_D rate without {feature_name}:  {100*train_err[-1]:.2f}")


# load_train_and_print_error(file_paths)
# plot_correlation_matrix(all_data)
def plot_step_size_convergence():
    step_sizes = [0.1, 0.01, 0.005, 0.001]   
    train_x, train_y, test_x, test_y, x, y = load_data()
    colors = ['b', 'g', 'r', 'c']
    for color, step_size in zip(colors, step_sizes):
        np.random.seed(0)
        c = classifier(3, len(train_x[0]))
        train_err, test_err = train_on_dataset(c, train_x, train_y, 2000, step_size, test_x, test_y)
        plt.plot(test_err, color=color, label=f"Step Size: {step_size}")#, alpha=0.2)
        # window_size = 10
        # running_avg = np.convolve(test_err, np.ones(window_size)/window_size, mode='valid')
        # plt.plot(running_avg, label=f"Step Size: {step_size}",color=color)
    plt.ylabel("$ERR_T$")
    plt.xlabel("Training step")
    plt.legend()
    plt.savefig("convergence.pdf", format="pdf")

# load_train_and_print_error(file_paths)

plot_step_size_convergence()
