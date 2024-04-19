import numpy as np
import data_loader
from matplotlib import pyplot as plt
from classifier import NN
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
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
    centroids[i*M:(i+1)*M] = cluster_centers[i*M:(i+1)*M].reshape(M, -1)


reshaped_centroids = centroids.reshape((10 * M, 28, 28))
for i in range(10):
    for j in range(M):
        plt.subplot(10, M, i * M + j + 1)
        plt.imshow(reshaped_centroids[i * M + j], cmap='gray')
        plt.axis('off')
plt.show()

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(cluster_centers, cluster_labels)

training_predictions = knn.predict(train_x)
test_predictions = knn.predict(test_x)

nn_conf_matrix_train = confusion_matrix(train_y, training_predictions)
nn_error_rate_train= 1 - accuracy_score(train_y, training_predictions)
print(" train Conf matrix with K = 1:\n", nn_conf_matrix_train)
print("train Error rate with K = 1:", nn_error_rate_train)
nn_conf_matrix = confusion_matrix(test_y, test_predictions)
nn_error_rate = 1 - accuracy_score(test_y, test_predictions)

print("Conf matrix with K = 1:\n", nn_conf_matrix)
print("Error rate with K = 1:", nn_error_rate)

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(cluster_centers, cluster_labels)
print("------------------------------------------\n")
test_predictions = knn.predict(test_x)
training_predictions = knn.predict(train_x)

nn_conf_matrix_train = confusion_matrix(train_y, training_predictions)
nn_error_rate_train= 1 - accuracy_score(train_y, training_predictions)
print(" train Conf matrix with K = 7:\n", nn_conf_matrix_train)
print("train Error rate with K = 7:", nn_error_rate_train)
knn_conf_matrix = confusion_matrix(test_y, test_predictions)
knn_error_rate = 1 - accuracy_score(test_y, test_predictions)

print("Conf Matrix K=7:\n", knn_conf_matrix)
print("Error Rate K=7):", knn_error_rate)

