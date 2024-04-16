
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d

def reduce_dataset(N, data_set):
    pca = PCA()
    pca.fit(data_set)
    
    print("pca explained variance", pca.explained_variance_ratio_)
    # Plot the explained variance ratio
    fig = plt.figure(1, figsize=(8, 6))
    plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.show()


# fig = plt.figure(1, figsize=(8, 6))
#     ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)
#     ax.scatter(
#     data_set_reduced[:, 0],
#     data_set_reduced[:, 1],
#     data_set_reduced[:, 2],
#     c=target,
#     s=40,
#     )

#     ax.set_title("First three PCA dimensions")
#     ax.set_xlabel("1st Eigenvector")
#     ax.xaxis.set_ticklabels([])
#     ax.set_ylabel("2nd Eigenvector")
#     ax.yaxis.set_ticklabels([])
#     ax.set_zlabel("3rd Eigenvector")
#     ax.zaxis.set_ticklabels([])

#     plt.show()