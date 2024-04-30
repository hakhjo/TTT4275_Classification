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

remove_features = ["petal length","sepal width", "sepal length"]
# remove_features = ["sepal width", "sepal length"]
# remove_features = ["sepal width"]
# remove_features = []
train_x, train_y, test_x, test_y, x, y = load_data(1, remove_features)
c = classifier(n_classes, n_features-len(remove_features))
train_err, test_err = train_on_dataset(c, train_x, train_y, 1500, 0.005, test_x, test_y)
test_conf = c.confusion(test_x, test_y)
train_conf = c.confusion(train_x, train_y)
# plot_confusion_matrix("test confusion matrix", test_conf)
# plot_confusion_matrix("training confusion matrix", train_conf)
# produce_histograms(x, y)
# plot_3d_decision_boundary_between_two_classes(c,train_x, train_t)
# plot_3d_decision_boundary_between_two_classes(c,test_x , test_t)
# plot_correlation_matrix(x)


display_results(train_err, test_err, train_conf, test_conf, len(train_x), len(test_x))

def load_train_and_print_error():
    train_x, train_y, test_x, test_y, x, y = load_data(drop_features="sepal width")
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
        print(len(test_err), len(test_x))
        l_test, u_test = wilson_CI(test_err[-1], len(test_x))
        l_train, u_train = wilson_CI(train_err[-1], len(train_x))
        print(f"EER_T without {feature_name}:  {100*test_err[-1]:.2f}, CI [{100*l_test:.2f},{100*u_test:.2f}] ")
        print(f"EER_D rate without {feature_name}:  {100*train_err[-1]:.2f}  CI [{100*l_train:.2f},{100*u_train:.2f}]")


def plot_step_size_convergence( train_x, train_y, test_x, text_y):
    step_sizes = [1.0,0.1, 0.01,0.001]   
    colors = ['b', 'g', 'r', 'c']
    for i, step_size in enumerate(step_sizes):
        c = classifier(3, len(train_x[0]))
        train_err, val_err = train_on_dataset(c, train_x, train_y, 2000, step_size, test_x, text_y)
        plt.plot(val_err, color=colors[i], alpha=0.2)
        window_size = 10
        running_avg = np.convolve(val_err, np.ones(window_size)/window_size, mode='valid')
        plt.plot(running_avg, label=f"$Step$ $Size:$ ${step_size}$",color=colors[i])
    plt.ylabel("$Error$ $rate$")
    plt.xlabel("$Iterations$")
    plt.legend()
    plt.savefig("convergence.pdf", format="pdf")

# pair_plot(x,y)
# load_train_and_print_error()

