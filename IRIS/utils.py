import numpy as np
from matplotlib import pyplot as plt
from classifier import *
from data import *

def display_results(train_err, test_err, train_conf, val_conf, len_train, len_test):
    val_conf = 100.0 * val_conf / np.sum(val_conf, axis=1)
    train_conf = 100.0 * train_conf / np.sum(train_conf, axis=1)
    l_test, u_test = wilson_CI(test_err[-1], len_test)
    l_train, u_train = wilson_CI(test_err[-1], len_train)


    print("\n------------- RESULTS ---------------")
    print("    Validation:            Training:")
    print(f"ERROR RATE: {100 * test_err[-1]:.2f}            {100 * train_err[-1]:.2f}")
    print(f"CI [{100*l_test:.2f},{100*u_test:.2f}]         CI [{100*l_train:.2f},{100*u_train:.2f}]")
    print("CONFUSION MATRICES")
    for vr, tr in zip(val_conf, train_conf):
        print(
            " ".join(f"{c:>6.2f}" for c in vr),
            " | ",
            " ".join(f"{c:>6.2f}" for c in tr),
        )

    plt.plot(train_err, "r", label="training error")
    plt.plot(test_err, "b", label="test error")
    plt.xlabel("Iteration")
    plt.ylabel("Error rate")
    plt.legend()
    plt.savefig("error_rate_20.png",format="png")


def wilson_CI(p0, n, z_a=1.6449):
    q0 = 1 - p0
    R = z_a * np.sqrt(p0 * q0 / n + z_a**2 / (4 * n**2)) / (1 + (z_a**2) / n)
    p = (p0 + (z_a**2) / (2 * n)) / (1 + z_a**2 / n)
    return p - R, p + R





def load_train_and_print_error():
    train_x, train_y, test_x, test_y, x, y = load_data()
    c = classifier(3, len(train_x[0]))
    train_err, test_err = c.train_on_dataset(train_x, train_y, 1500, 0.005, test_x, test_y)
    print(f"ERR_T with all features: {100*test_err[-1]:.2f}")
    print(f"ERR_D with all features: {100*train_err[-1]:.2f}")

    for feature_name in feature_names:
        np.random.seed(0)
        train_x, train_y, test_x, test_y, x, y = load_data(drop_features=feature_name)
        c = classifier(3, len(train_x[0]))
        train_err, test_err = c.train_on_dataset(train_x, train_y, 1500, 0.005, test_x, test_y)
        print(len(test_err), len(test_x))
        l_test, u_test = wilson_CI(test_err[-1], len(test_x))
        l_train, u_train = wilson_CI(train_err[-1], len(train_x))
        print(f"EER_T without {feature_name}:  {100*test_err[-1]:.2f}, CI [{100*l_test:.2f},{100*u_test:.2f}] ")
        print(f"EER_D rate without {feature_name}:  {100*train_err[-1]:.2f}  CI [{100*l_train:.2f},{100*u_train:.2f}]")

