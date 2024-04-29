import numpy as np
from matplotlib import pyplot as plt

def display_results(train_err, test_err, train_conf, val_conf):
    val_conf = 100.0 * val_conf / np.sum(val_conf, axis=1)
    train_conf = 100.0 * train_conf / np.sum(train_conf, axis=1)
    print("\n------------- RESULTS ---------------")
    print("    Validation:            Training:")
    print(f"ERROR RATE: {100 * test_err[-1]:.2f}            {100 * train_err[-1]:.2f}")
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


def wilson_CI(p0, n, z_a=1.96):
    q0 = 1 - p0
    R = z_a * np.sqrt(p0 * q0 / n + z_a**2 / (4 * n**2)) / (1 + (z_a**2) / n)
    p = (p0 + (z_a**2) / (2 * n)) / (1 + z_a**2 / n)
    return p - R, p + R
