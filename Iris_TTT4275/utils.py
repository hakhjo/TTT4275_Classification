import numpy as np
from matplotlib import pyplot as plt

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
    plt.savefig("error_rate_20.pdf",format="pdf")