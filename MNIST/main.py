import numpy as np
from tqdm import tqdm, trange
import utils
from matplotlib import pyplot as plt
from classifier import NNClassifier, ClusteredKNNClassifier
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import pickle

np.random.seed(0)
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = """\\usepackage{bm}\n\\usepackage{upgreek}"""

train_x, train_y, test_x, test_y = utils.load_data()

M = 64
N = 10


def run_KNN_timed(c: ClusteredKNNClassifier, K):
    start = time.time()
    test_predictions = c.predict_array(test_x, K)
    test_time = time.time() - start
    res = utils.Result(
        test_predictions, test_y[:, 0], test_time, N, f"{K}-NN, entire test set"
    )
    res.report()
    res.save(f"{K}-NN_cmplt")
    return res


def run_NN_timed(c: NNClassifier):
    start = time.time()
    test_predictions = c.predict_array(test_x, chunk_size=1000)
    test_time = time.time() - start
    res = utils.Result(
        test_predictions, test_y[:, 0], test_time, N, "NN, entire test set"
    )
    res.report()
    res.save("NN_cmplt")
    return res


# res = {"err":[], "K":[]}
# for K in range(1, 21):
#     c = KNeighborsClassifier(n_neighbors=K)
#     c.fit(train_x, train_y[:, 0])
#     guesses = c.predict(test_x)
#     err = utils.error_rate(guesses, test_y[:, 0])
#     res["err"].append(err)
#     res["K"].append(K)
#     print(K, err)

# with open("K-test", "wb") as f:
#     pickle.dump(res, f)


def k_comparison():

    plt.rc("axes", titlesize=14)  # fontsize of the axes title
    plt.rc("axes", labelsize=14)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=14)

    with open("K-test", "rb") as f:
        res = pickle.load(f)
    l, u = utils.wilson_CI(np.array(res["err"]), len(test_x))
    plt.errorbar(
        res["K"],
        res["err"],
        [res["err"] - l, u - res["err"]],
        capsize=5,
        fmt="r.",
        ecolor="k",
        elinewidth=0.8,
        capthick=0.8,
    )
    plt.xlabel("$k$")
    plt.ylabel("$p_e$")
    plt.xticks(list(range(res["K"][0], res["K"][-1] + 2, 2)))
    plt.savefig("figs/k-comparison.pdf")
    # plt.fill_between(res["K"], l, u, alpha=0.1)
    plt.show()


def explore_predictions():
    c = NNClassifier(train_x, train_y, chunk_size=1000)
    for idx, (x, y) in enumerate(zip(test_x[500:], test_y[500:])):
        g = c.predict_single(x)
        if g != y[0]:
            print(f"{idx=}: g={g}, y={y[0]}")
            # utils.display_image(x)
            plt.imshow(x.reshape((28, 28)), cmap="Greys")
            plt.show()

def clust_explore_predictions():
    c = ClusteredKNNClassifier(train_x, train_y, 10, 64)
    for idx, (x, y) in enumerate(zip(test_x[300:], test_y[300:])):
        g = c.predict_single(x, 1)
        if g != y[0]:
            print(f"{300+idx=}: g={g}, y={y[0]}")
            # utils.display_image(x)
            plt.imshow(x.reshape((28, 28)), cmap="Greys")
            plt.show()


def save_examples():
    c = NNClassifier(train_x, train_y, chunk_size=1000)
    bad_idcs = [-96, -117, -150]
    good_idcs = [500, 501, 502]
    plt.rc("axes", titlesize=40)
    for n, i in enumerate(bad_idcs):
        plt.imshow(test_x[i].reshape((28, 28)), cmap="Greys")
        plt.xticks([])
        plt.yticks([])
        plt.title(
            f"$\hat\omega={c.predict_single(test_x[i])[0]}$, $\omega={test_y[i][0]}$"
        )
        plt.savefig(f"figs/mis{n+1}.pdf", bbox_inches="tight", pad_inches=0.5)
    for n, i in enumerate(good_idcs):
        plt.imshow(test_x[i].reshape((28, 28)), cmap="Greys")
        plt.title(
            f"$\hat\omega={c.predict_single(test_x[i])[0]}$, $\omega={test_y[i][0]}$"
        )
        plt.savefig(f"figs/good{n+1}.pdf", bbox_inches="tight", pad_inches=0.5)


def disp_NN():
    idcs = [-117, -150, -400, -502]
    plt.rc("figure", titlesize=30)
    plt.rc("axes", titlesize=40)
    plt.rc("font", size=35)
    c = NNClassifier(train_x, train_y, chunk_size=1000)
    for n, i in enumerate(tqdm(idcs)):
        g, nn = c.get_nn(test_x[i])
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(test_x[i].reshape((28, 28)), cmap="Greys")
        ax1.text(1.5, 4, "$\mathbf{{x}}$")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.imshow(nn.reshape((28, 28)), cmap="Greys")
        ax2.text(1.5, 4, "$\\underline{{\mathbf{{x}}}}_{{n^\\ast}}$")
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax1.set_title(f"$\omega={test_y[i][0]}$", y=-0.25)
        ax2.set_title(f"$\hat \omega={g[0]}$", y=-0.25)
        fig.suptitle(f"Test sample \#{len(test_x)+i+1}", y=0.78)
        fig.tight_layout()
        # plt.show()
        plt.savefig(f"figs/cmp{n+1}.pdf", bbox_inches="tight", pad_inches=0.2)

def disp_CNN():
    idcs = [320, 335, 305, 310]
    plt.rc("figure", titlesize=30)
    plt.rc("axes", titlesize=40)
    plt.rc("font", size=35)
    c = ClusteredKNNClassifier(train_x, train_y, 10, 64)
    for n, i in enumerate(tqdm(idcs)):
        g, nn = c.get_nn(test_x[i])
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(test_x[i].reshape((28, 28)), cmap="Greys")
        ax1.text(1.5, 4, "$\mathbf{{x}}$")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.imshow(nn.reshape((28, 28)), cmap="Greys")
        ax2.text(1.5, 4, "$\\bm{{\\upmu}}_{{n^\\ast}}$")
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax1.set_title(f"$\omega={test_y[i][0]}$", y=-0.25)
        ax2.set_title(f"$\hat \omega={g}$", y=-0.25)
        fig.suptitle(f"Test sample \#{i+1}", y=0.78)
        fig.tight_layout()
        # plt.show()
        plt.savefig(f"figs/clust_cmp{n+1}.pdf", bbox_inches="tight", pad_inches=0.2)

def print_table():
    start_cluster = time.time()
    c = ClusteredKNNClassifier(train_x, train_y, N, M)
    cluster_time = time.time() - start_cluster

    print(f"Clustering took {cluster_time}s")

    res = utils.result_from_file("NN_cmplt")
    pl, ph = utils.wilson_CI(res.err, len(test_x))
    print(
        f"{100*res.err} & $\\left\\[{100*pl},\\;{100*ph}\\right\\] & {res.time//60:.0f}:{res.time%60:.1f}"
    )

    res = utils.result_from_file("1-NN_cmplt")
    pl, ph = utils.wilson_CI(res.err, len(test_x))
    print(
        f"{100*res.err} & $\\left\\[{100*pl},\\;{100*ph}\\right\\] & 00:{cluster_time+res.time:.1f}"
    )

    res = utils.result_from_file("7-NN_cmplt")
    pl, ph = utils.wilson_CI(res.err, len(test_x))
    print(
        f"{100*res.err} & $\\left\\[{100*pl},\\;{100*ph}\\right\\] & 00:{cluster_time+res.time:.1f}"
    )


def test_k():
    res = {"k": [], "score": [], "time": []}
    for k in tqdm(range(1, 129)):
        score = 0
        t0 = time.time()
        for i in range(10):
            indices = np.where(train_y == i)[0]
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(train_x[indices])
            score -= kmeans.score(train_x[indices])
        res["time"].append(time.time()-t0)
        res["score"].append(score)
        res["k"].append(k)

    with open("kmeans-test", "wb") as f:
        pickle.dump(res, f)

def plot_test_k():
    with open("kmeans-test", "rb") as f:
        res = pickle.load(f)
        l1 = plt.plot(res["k"], res["score"], label="$J^\\ast$")
        plt.tick_params(axis="y", labelcolor="b", color="b", labelsize=14)
        plt.tick_params(axis="x", labelsize=14)
        plt.twinx()
        l2 = plt.plot(res["k"], res["time"], label="clustering time", color="r")
        plt.xlabel("K", fontsize=14)
        plt.ylabel("[s]", color="r", fontsize=14)
        plt.tick_params(axis="y", labelcolor="r", color="r", labelsize=14)
        lines = l1+l2
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels, loc=9, fontsize=14)
        plt.savefig("figs/kmeans_k_comparison.pdf", bbox_inches="tight")
        plt.show()
    
# clust_explore_predictions()
# res = utils.result_from_file("NN_cmplt")
# utils.plot_confusion_matrix(res.conf)
# disp_CNN()

plot_test_k()


# misclassifications()

# c = NNClassifier(train_x, train_y, chunk_size=1000)
# run_NN_timed(c)


# run_KNN_timed(c, 1)
# run_KNN_timed(c, 7)

# disp_NN()

