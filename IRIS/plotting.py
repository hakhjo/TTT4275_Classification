from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import numpy as np
from scipy.stats import norm, pearsonr
from data import *
from classifier import classifier


plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["text.usetex"] = True


def plot_histograms(x, y):
    class_labels = np.argmax(y, axis=1)
    mean_features = np.zeros((n_classes, n_features))
    for i in range(n_classes):
        mean_features[i, :] = np.mean(x[class_labels == i], axis=0)

    fig, axs = plt.subplots(n_features, 1, figsize=(14, 10))

    for i in range(n_features):
        feature_min = np.min(x[:, i])
        feature_max = np.max(x[:, i])
        bins = np.arange(feature_min, feature_max + 0.1, 0.1)
        x_values = np.linspace(feature_min, feature_max, 300)
        for j in range(n_classes):
            feature_values = x[class_labels == j, i]
            mean = np.mean(feature_values)
            std = np.std(feature_values)

            axs[i].hist(
                feature_values,
                bins=bins,
                alpha=0.4,
                label=class_names[j],
                density=True,
                edgecolor="black",
            )
            normal_dist = norm.pdf(x_values, mean, std)
            axs[i].plot(x_values, normal_dist)

        for mean in mean_features[:, i]:
            axs[i].axvline(x=mean, color="r", linestyle="dashed", linewidth=1)

        axs[i].set_title(feature_names[i], fontsize=25)
        axs[i].tick_params(axis="both", which="major", labelsize=20)
        if i == 1:
            axs[i].legend(fontsize=20)
    plt.tight_layout()
    plt.savefig("features_IRIS.pdf", format="pdf")


def plot_confusion_matrix(name, conf_mat):
    df_cm = pd.DataFrame(
        conf_mat,
        index=[i for i in class_names_short],
        columns=[i for i in class_names_short],
    )

    g = sn.heatmap(
        df_cm, annot=True, annot_kws={"size": 26}, cbar=False, square=True, fmt="g"
    )
    g.set_xticklabels([i for i in class_names_short], fontsize=18)
    g.set_yticklabels([i for i in class_names_short], fontsize=18)
    plt.xlabel("$\\hat{\\omega}$", fontsize=20)
    plt.ylabel("$\\omega$", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{name}.pdf", format="pdf")
    plt.clf()


def pair_plot(x, y):
    df = pd.DataFrame(x, columns=feature_names)
    t = np.argmax(y, axis=1)
    sn.set_theme(
        font_scale=2,
        rc={
            "text.usetex": True,
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
        },
    )
    g = sn.PairGrid(df)

    for i, j in zip(*np.triu_indices_from(g.axes, k=1)):
        ax = g.axes[i, j]
        r, _ = pearsonr(df.iloc[:, i], df.iloc[:, j])
        ax.annotate(
            f"$\\rho$ = {r:.2f}", xy=(0.5, 0.5), xycoords="axes fraction", ha="center"
        )
        ax.set_axis_off()

    for i in range(len(feature_names)):
        ax = g.axes[i, i]
        ax.clear()
        ax.text(
            0.5,
            0.5,
            feature_names[i],
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=20,
        )
        ax.set_axis_off()

    g.hue_names = class_names
    g.map_lower(sn.scatterplot, hue=[class_names[i] for i in t])
    g.add_legend()
    g.set(xlabel="", ylabel="")
    plt.savefig("scatter_plots.pdf", format="pdf")


def plot_step_size_convergence(train_x, train_y, test_x, test_y):
    step_sizes = [1.0, 0.1, 0.01, 0.001]
    colors = ["b", "g", "r", "c"]
    for color, step_size in zip(colors, step_sizes):
        np.random.seed(0)
        c = classifier(3, len(train_x[0]))
        train_err, val_err = c.train_on_dataset(
            train_x, train_y, 1500, 0.005, test_x, test_y
        )
        plt.plot(val_err, color=colors[i], alpha=0.2)
        window_size = 10
        running_avg = np.convolve(
            val_err, np.ones(window_size) / window_size, mode="valid"
        )
        plt.plot(running_avg, label=f"$Step$ $Size:$ ${step_size}$", color=colors[i])
    plt.ylabel("$Error$ $rate$")
    plt.xlabel("$Iterations$")
    plt.legend()
    plt.savefig("convergence.pdf", format="pdf")
