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




def k_comparison():
    plt.figure(figsize=(6,4.5))
    plt.rc("axes", titlesize=20)  # fontsize of the axes title
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
    plt.yticks([0.028, 0.032, 0.036, 0.04])
    plt.title("Training set as templates")
    plt.tight_layout()
    plt.savefig("figs/k-comparison.pdf")#, bbox_inches="tight")
    # plt.fill_between(res["K"], l, u, alpha=0.1)
    # plt.show()


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
    res = {"k": [], "score": [], "time": [], "err": []}
    for k in tqdm(range(1, 129)):
        score = 0
        cluster_centers = np.zeros((10 * k, train_x.shape[1]))
        cluster_labels = np.zeros((10 * k,), dtype=int)
        t = 0
        for i in range(10):
            indices = np.where(train_y == i)[0]
            kmeans = KMeans(n_clusters=k)
            t0 = time.time()
            kmeans.fit(train_x[indices])
            t += time.time()-t0
            score -= kmeans.score(train_x[indices])
            cluster_centers[i * k : (i + 1) * k] = (
                kmeans.cluster_centers_
            )
            cluster_labels[i * k : (i + 1) * k] = i
        c = KNeighborsClassifier(n_neighbors=1)
        c.fit(cluster_centers, cluster_labels)
        pred = c.predict(test_x)
        err = utils.error_rate(pred, test_y[:, 0])
        res["time"].append(t)
        res["score"].append(score)
        res["err"].append(err)
        res["k"].append(k)

    with open("kmeans-test2", "wb") as f:
        pickle.dump(res, f)


def plot_test_kmeans2():
    with open("kmeans-test2", "rb") as f:
        fig = plt.figure(layout='constrained')

        res = pickle.load(f)
        l, u = utils.wilson_CI(np.array(res["err"]), len(test_x))
        
        ax1 = plt.gca()
        ax2 = plt.twinx()
        ax3 = plt.twinx()

        l1 = ax1.plot(res["k"], res["score"], label="$J^\\ast$", color="b")
        ax1.tick_params(axis="y", labelcolor="b", color="b", labelsize=14)
        ax1.tick_params(axis="x", labelsize=14)
        ax1.set_ylabel("$J$", color="b", fontsize=14)
        ax1.set_xlabel("$K$", fontsize=14)

        l2 = ax2.plot(res["k"], res["time"], label="clustering time", color="r")
        ax2.set_ylabel("time [s]", color="r", fontsize=14)
        ax2.tick_params(axis="y", labelcolor="r", color="r", labelsize=14)
        ax2.yaxis.set_label_position("right")
        ax2.yaxis.set_ticks_position("right")

        l3 = ax3.plot(res["k"], res["err"], label="$ERR_T$", color="g")
        ax3.fill_between(res["k"], l, u, color="g", alpha=0.2)
        ax3.set_ylabel("$p_e$", color="g", fontsize=14)
        ax3.tick_params(axis="y", labelcolor="g", color="g", labelsize=14)
        ax3.spines["left"].set_position(('outward', 60))
        ax3.yaxis.set_label_position("left")
        ax3.yaxis.set_ticks_position("left")
        ax3.spines["left"].set_visible(True)
        ax3.spines["right"].set_visible(False)

        lines = l1+l2+l3
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels, loc=9, fontsize=14)
        # plt.tight_layout()
        plt.savefig("figs/kmeans_k_comparison2.pdf")
        plt.show()
        
def plot_test_k():
    with open("kmeans-test", "rb") as f:
        res = pickle.load(f)
        l1 = plt.plot(res["k"], res["score"], label="$J^\\ast$", color="b")
        plt.tick_params(axis="y", labelcolor="b", color="b", labelsize=14)
        plt.tick_params(axis="x", labelsize=14)
        plt.xlabel("K", fontsize=14)
        plt.twinx()
        l2 = plt.plot(res["k"], res["time"], label="clustering time", color="r")
        plt.ylabel("[s]", color="r", fontsize=14)
        plt.tick_params(axis="y", labelcolor="r", color="r", labelsize=14)
        lines = l1+l2
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels, loc=9, fontsize=14)
        plt.savefig("figs/kmeans_k_comparison.pdf", bbox_inches="tight")
        plt.show()
    
# clust_explore_predictions()
def plot_confusion():
    for fn in tqdm(("NN", "1-NN", "7-NN")):
        res = utils.result_from_file(f"{fn}_cmplt")
        utils.plot_confusion_matrix(utils.confusion_percentage(res.conf), file=f"figs/{fn}_conf_p.pdf")


def clust_test_k():
    res = {"err":[], "K":[], "d":[]}
    c = ClusteredKNNClassifier(train_x, train_y, 10, 64)
    for K in range(1, 21):
        guesses = c.predict_array(test_x, K)
        err = utils.error_rate(guesses, test_y[:, 0])
        res["err"].append(err)
        res["K"].append(K)
        print(K, err)

    with open("clust-K-test", "wb") as f:
        pickle.dump(res, f)


def clust_test_k2():
    with open("clust-K-test", "rb") as f:
        res = pickle.load(f)
    res["d"] = []
    c = ClusteredKNNClassifier(train_x, train_y, 10, 64)
    for K in range(1, 21):
        dist = c.predict_array(test_x, K)
        res["d"].append(dist)
        print(K, dist)

    with open("clust-K-test2", "wb") as f:
        pickle.dump(res, f)



def test_k2():
    with open("K-test", "rb") as f:
        res = pickle.load(f)
    res["d"] = []
    c = KNeighborsClassifier()
    c.fit(train_x, train_y[:, 0])
    distmat, _ = c.kneighbors(test_x, n_neighbors=20)
    for K in range(1, 21):
        dist = np.mean(np.sqrt(distmat[:, K-1]))
        res["d"].append(dist)
        print(K, dist)

    with open("K-test2", "wb") as f:
        pickle.dump(res, f)

    

def plot_clust_k():
    plt.figure(figsize=(6,4.5))
    plt.rc("axes", titlesize=20)  # fontsize of the axes title
    plt.rc("axes", labelsize=14)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=14)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=14)

    with open("clust-K-test", "rb") as f:
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
    plt.title("$K$-means clustered templates, $K=64$")
    plt.tight_layout()
    plt.savefig("figs/cknn-k-comparison.pdf")#, bbox_inches="tight")
    # plt.fill_between(res["K"], l, u, alpha=0.1)
    # plt.show()


def plot_clust_k2():
    plt.figure(figsize=(6,4.5))
    plt.rc("axes", titlesize=24)  # fontsize of the axes title
    plt.rc("axes", labelsize=20)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=18)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=18)

    with open("clust-K-test2", "rb") as f:
        res = pickle.load(f)
    l, u = utils.wilson_CI(np.array(res["err"]), len(test_x))
    l1 = plt.errorbar(
        res["K"],
        res["err"],
        [res["err"] - l, u - res["err"]],
        capsize=5,
        fmt="r.",
        ecolor="r",
        elinewidth=0.8,
        capthick=0.8,
        label="$ERR_T$",
    )
    plt.xlabel("$k$")
    # plt.ylabel("$p_e$")
    plt.tick_params(axis="y", labelcolor="r", color="r")
    plt.twinx()
    l2 = plt.scatter(res["K"], res["d"], 10, label="$\\overline{\sqrt{d_k}}$", color="b")
    plt.xticks(list(range(res["K"][0], res["K"][-1] + 2, 2)))
    plt.tick_params(axis="y", labelcolor="b", color="b")
    plt.title("$K$-means clustered templates, $K=64$")
    lines = (l2, l1)
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, fontsize=16)
    plt.tight_layout()
    plt.savefig("figs/cknn-k-comparison2.pdf")#, bbox_inches="tight")
    # plt.show()

def plot_test_k2():
    plt.figure(figsize=(6,4.5))
    plt.rc("axes", titlesize=24)  # fontsize of the axes title
    plt.rc("axes", labelsize=20)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=18)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=18)

    with open("K-test2", "rb") as f:
        res = pickle.load(f)
    l, u = utils.wilson_CI(np.array(res["err"]), len(test_x))
    l1 = plt.errorbar(
        res["K"],
        res["err"],
        [res["err"] - l, u - res["err"]],
        capsize=5,
        fmt="r.",
        ecolor="r",
        elinewidth=0.8,
        capthick=0.8,
        label="$ERR_T$",
    )
    plt.xlabel("$k$")
    # plt.ylabel("$p_e$")
    plt.tick_params(axis="y", labelcolor="r", color="r")
    plt.yticks([0.028, 0.032, 0.036, 0.04])
    plt.twinx()
    l2 = plt.scatter(res["K"], res["d"], 10, label="$\\overline{\sqrt{d_k}}$", color="b")
    plt.xticks(list(range(res["K"][0], res["K"][-1] + 2, 2)))
    plt.tick_params(axis="y", labelcolor="b", color="b")
    plt.title("Training set as templates")
    lines = (l2, l1)
    labels = [l.get_label() for l in lines]
    # plt.legend(lines, labels, fontsize=16)
    plt.tight_layout()
    plt.savefig("figs/k-comparison2.pdf")#, bbox_inches="tight")


# clust_test_k2()
# plot_test_k2()
# plot_clust_k2()
# plot_test_k2()
plot_confusion()
# print(len(train_x))
# k_comparison()
# plot_clust_k()
# disp_CNN()

# plot_confusion()
# test_k()
# plot_test_kmeans2()
# plot_test_k()


# misclassifications()

# c = NNClassifier(train_x, train_y, chunk_size=1000)
# run_NN_timed(c)


# run_KNN_timed(c, 1)
# run_KNN_timed(c, 7)

# disp_NN()

