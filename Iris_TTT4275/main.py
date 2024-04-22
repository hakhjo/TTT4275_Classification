import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from classifier import classifier 
from data_modification import *
from matplotlib import pyplot as plt
import seaborn as sn

encoder = OneHotEncoder(sparse_output=False, categories="auto")
class_name = ["Iris Setosa","Iris Versicolour", "Iris Virginica"]
class_labels = np.array([[1], [2], [3]])
encoder.fit(class_labels)
np.random.seed(0)


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


def train_on_dataset(c: classifier, x, t, N, step_size, x_val, t_val):
    train_err = np.zeros(N)
    val_err = np.zeros(N)
    assert len(x) == len(t)
    for i in range(N):
        p = np.random.permutation(len(x))
        err = c.train(x[p, :], t[p, :], step_size)
        train_err[i] = err
        val_err[i] = c.validate(x_val, t_val)
        print(f"TRAINING... {i}/{N}: \t{100*err:.2f}", end="\r", flush=True)

    print("TRAINING... DONE                      ")
    return train_err, val_err


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


file_paths = ["class_1", "class_2", "class_3"]
train_x, train_t, val_x, val_t = load_and_process_data(file_paths)
# produce_histograms_split(file_paths)
# plot_feature_trends(file_paths)
N = 3
train_x = reduce_dataset(N, train_x)
val_x = reduce_dataset(N, val_x)
# plot_dataset(N, train_x, train_t)
c = classifier(3, N)
train_err, val_err = train_on_dataset(
    c, train_x, train_t, 2000, 0.001,train_x, train_t
)
val_conf = c.confusion(val_x, val_t)
train_conf = c.confusion(train_x, train_t)
plot_confusion_matrix("validation confusion matrix", val_conf)
plot_confusion_matrix("training confusion matrix", train_conf)
# display_results(train_err, val_err, train_conf, val_conf)

def plot_step_size_convergence():
    step_sizes = [0.001, 0.01, 0.1, 1.0]   
    colors = ['b', 'g', 'r', 'c']
    for i, step_size in enumerate(step_sizes):
        c = classifier(3, N)
        train_err, val_err = train_on_dataset(c, train_x, train_t, 2000, step_size, train_x, train_t)
        plt.plot(val_err, color=colors[i], alpha=0.2)
        window_size = 10
        running_avg = np.convolve(val_err, np.ones(window_size)/window_size, mode='valid')
        plt.plot(running_avg, label=f"Step Size: {step_size}",color=colors[i])
    plt.ylabel("Error rate")
    plt.xlabel("Iterations")
    plt.legend()
    plt.savefig("convergence.pdf", format="pdf")

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

def plot_3d_decision_boundary_between_two_classes(classifier, X, t):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define ranges and meshgrid
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    x3_min, x3_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    x1_range = np.linspace(x1_min, x1_max, num=100)
    x2_range = np.linspace(x2_min, x2_max, num=100)
    x3_range = np.linspace(x3_min, x3_max, num=100)
    x1, x2 = np.meshgrid(x1_range, x2_range)
    
    w = classifier.W[1, :] - classifier.W[2, :]
    x3 = -(w[0] * x1 + w[1] * x2 + w[3]) / w[2]
    
    # Plot the decision boundary surface
    surf = ax.plot_surface(x1, x2, x3, color='magenta', alpha=0.3)
    classes_to_plot = [1,2]
    labels = np.argmax(t, axis=1)
    class_colors = np.array(['r', 'g', 'b'])
    for c in classes_to_plot:
        ix = np.where(labels == c)
        ax.scatter(X[ix, 0], X[ix, 1], X[ix, 2], c=class_colors[c], label=f'{class_name[c]}')
    
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_zlim(x3_min, x3_max)
    ax.set_title(f'3D Decision Boundary Between {class_name[1]} and {class_name[2]}')
    surf_legend = Line2D([0], [0], linestyle="none", c='magenta', marker = 'o')
    ax.legend([surf_legend] + [ax.scatter([],[],[], color=class_colors[c], label=f'{class_name[c]}', edgecolor='k') for c in classes_to_plot],
              ['Decision boundary'] + [f'{class_name[c]}' for c in classes_to_plot], numpoints=1)


    plt.show()


plot_3d_decision_boundary_between_two_classes(c, train_x, train_t)