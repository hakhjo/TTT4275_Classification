import numpy as np
import data_loader
from matplotlib import pyplot as plt
from classifier import NN
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import time
def display_image(vec):
    plt.imshow(vec.reshape((28, 28)))
    plt.show()

train_x, train_y, test_x, test_y = data_loader.load_data()
# print(train_x.shape)
# p = np.random.permutation(len(train_x))[:1000]
# c = NN(train_x[p], train_y[p], 10, 784)
# test_conf, err = c.confusion(test_x, test_y)
# train_conf = c.confusion(train_x, train_y)
# test_conf = 100.0 * test_conf / np.sum(test_conf, axis=1)
# # train_conf = 100.0 * train_conf / np.sum(train_conf, axis=1)
# print(f"Error rate: {err*100:.2f}")
# for tr in test_conf:
#     print(" ".join(f"{c:>6.2f}" for c in tr))
# print(c.evaluate(test_x[0]), test_y[0])
# display_image(test_x[0])



M = 64
kmeans = KMeans(n_clusters=64)
cluster_centers = np.zeros((10 * M, train_x.shape[1]))
cluster_labels = np.zeros((10 * M,), dtype=int)
centroids = np.zeros((10 * M, train_x.shape[1]))
for i in range(10):
    indices = np.where(train_y == i)[0]
    kmeans = KMeans(n_clusters=M)
    kmeans.fit(train_x[indices])
    cluster_centers[i*M:(i+1)*M] = kmeans.cluster_centers_
    cluster_labels[i*M:(i+1)*M] = i   

def plot_centroids(cluster_centers):
    for i in range(10):
        centroids[i*M:(i+1)*M] = cluster_centers[i*M:(i+1)*M].reshape(M, -1)
    reshaped_centroids = centroids.reshape((10 * M, 28, 28))
    for i in range(10):
        for j in range(M):
            plt.subplot(10, M, i * M + j + 1)
            plt.imshow(reshaped_centroids[i * M + j], cmap='gray')
            plt.axis('off')
    plt.show()

def KNearestNeighbours(n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    knn.fit(cluster_centers, cluster_labels)
    start = time.time()
    test_predictions = knn.predict(test_x)
    end = time.time()
    print(f"Time used NN with {n_neighbors}: {end - start}")
    training_predictions = knn.predict(train_x)

    nn_conf_matrix_train = confusion_matrix(train_y, training_predictions)
    nn_error_rate_train= 1 - accuracy_score(train_y, training_predictions)
    print(f" train Conf matrix with K = {n_neighbors}:\n {nn_conf_matrix_train}")
    print(f"train Error rate with K = {n_neighbors}:{nn_error_rate_train}")
    nn_conf_matrix = confusion_matrix(test_y, test_predictions)
    nn_error_rate = 1 - accuracy_score(test_y, test_predictions)

    print(f"Conf matrix with K = {n_neighbors}:\n {nn_conf_matrix}")
    print(f"Error rate with K = {n_neighbors} {nn_error_rate}")

    return end -start
# plot_centroids(cluster_centers)
time_1 = KNearestNeighbours(1)
time_7 = KNearestNeighbours(7)
print("increas in time:", time_7/time_1)


