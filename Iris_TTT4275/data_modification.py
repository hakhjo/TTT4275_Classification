from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import mpl_toolkits.mplot3d
import seaborn as sn
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, categories="auto")

class_labels = np.array([[1], [2], [3]])
encoder.fit(class_labels)

def reduce_dataset(N, data_set, plot=False):
    pca = PCA(n_components=N)
    data_set_reduced = pca.fit_transform(data_set)
    if plot:
        fig = plt.figure(1, figsize=(8, 6))
        plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.show()
    
    print("shape of dataset:", data_set_reduced.shape)
    return data_set_reduced

def plot_dataset(N, x_reduced, t_reduced):    
    if N == 3:
        fig = plt.figure(1, figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        colors = ['r', 'g', 'b']  # Define colors for each class
        for i in range(len(x_reduced)):
            class_index = np.argmax(t_reduced[i])
            ax.scatter(x_reduced[i, 0], x_reduced[i, 1], x_reduced[i, 2], c=colors[class_index])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.legend()
        plt.show()
    
feature_names = ["sepal length","sepal width"," petal length","petal width" ]
class_name = ["Iris Setosa","Iris Versicolour", "Iris Virginica"]

def produce_histograms(x, t):
    class_labels = np.argmax(t, axis=1)
    num_classes = t.shape[1]
    num_features = x.shape[1]
    
    mean_features = np.zeros((num_classes, num_features))
    
    for i in range(num_classes):
        mean_features[i, :] = np.mean(x[class_labels == i], axis=0)
    
    print(mean_features)
    
    fig, axs = plt.subplots(num_features, 1, figsize=(10, 8))
    
    for i in range(num_features):
        feature_min = np.min(x[:, i])
        feature_max = np.max(x[:, i])
        bins = np.arange(feature_min, feature_max + 0.1, 0.1)  
        x_values = np.linspace(feature_min, feature_max, 300)
        for j in range(num_classes):
            feature_values = x[class_labels == j, i]
            mean = np.mean(feature_values)
            std = np.std(feature_values)
            print(mean, std)
            
            axs[i].hist(feature_values, bins=bins, alpha=0.4, label=class_name[j], density=True, edgecolor='black')
            normal_dist = norm.pdf(x_values, mean, std)
            axs[i].plot(x_values, normal_dist, label=f'{class_name[j]} PDF')
        
        for mean in mean_features[:, i]:
            axs[i].axvline(x=mean, color='r', linestyle='dashed', linewidth=1)
        
        axs[i].set_title(feature_names[i])
        axs[i].legend()
    
    plt.tight_layout()
    plt.show()


def produce_histograms_split(file_paths):
    data = []
    labels = []
    
    # Example of predefined variables
    encoder = {i: i-1 for i in range(1, len(file_paths)+1)}  # Simple index mapping i to i-1
    feature_names = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4']  # Example feature names
    
    for i, file_path in enumerate(file_paths, start=1):
        read = pd.read_csv(file_path, header=None)
        data.append(read)
        class_index = encoder[i]  # Use simple index mapping
        labels.append(np.full((read.shape[0],), class_index))  # Create label array filled with class_index
    
    data = pd.concat(data, ignore_index=True).to_numpy()
    labels = np.concatenate(labels)
    num_features = data.shape[1]
    num_classes = len(file_paths)

    # Calculate mean features per class
    mean_features = np.zeros((num_classes, num_features))
    for i in range(num_classes):
        mean_features[i, :] = np.mean(data[labels == i, :], axis=0)

    fig, axs = plt.subplots(num_features, 2, figsize=(15, 8))

    for i in range(num_features):
        for j in range(num_classes):
            class_data = data[labels == j, i]
            feature_values_first = class_data[:25]
            feature_values_last = class_data[-25:]

            # Calculate means and standard deviations for normal distributions
            mean_first = np.mean(feature_values_first)
            std_first = np.std(feature_values_first)
            mean_last = np.mean(feature_values_last)
            std_last = np.std(feature_values_last)

            feature_min = min(feature_values_first.min(), feature_values_last.min())
            feature_max = max(feature_values_first.max(), feature_values_last.max())
            bins = np.linspace(feature_min, feature_max, 30)
            x_values = np.linspace(feature_min, feature_max, 300)

            # Plot histograms and normal PDFs
            if j == 0:  # Only plot for the first class to initialize the plots
                axs[i, 0].hist(feature_values_first, bins=bins, alpha=0.4, color='blue', density=True)
                axs[i, 1].hist(feature_values_last, bins=bins, alpha=0.4, color='green', density=True)
            else:
                axs[i, 0].hist(feature_values_first, bins=bins, alpha=0.4, color='blue', density=True)
                axs[i, 1].hist(feature_values_last, bins=bins, alpha=0.4, color='green', density=True)

            normal_dist_first = norm.pdf(x_values, mean_first, std_first)
            normal_dist_last = norm.pdf(x_values, mean_last, std_last)
            axs[i, 0].plot(x_values, normal_dist_first, 'r--', label=f'{class_name[j]}: μ={mean_first:.2f}, σ={std_first:.2f}')
            axs[i, 1].plot(x_values, normal_dist_last, 'r--', label=f'{class_name[j]}: μ={mean_last:.2f}, σ={std_last:.2f}')

            # Plotting the mean lines for reference
            axs[i, 0].axvline(x=mean_first, color='red', linestyle='dashed', linewidth=1)
            axs[i, 1].axvline(x=mean_last, color='red', linestyle='dashed', linewidth=1)

        axs[i, 0].set_title(f'{feature_names[i]} - First 25 Samples')
        axs[i, 1].set_title(f'{feature_names[i]} - Last 25 Samples')
        axs[i, 0].legend()
        axs[i, 1].legend()

    plt.tight_layout()
    plt.show()

def plot_feature_trends(file_paths):
    num_features = 4  # Assuming there are 4 features per file

    # Process each file
    for file_path in file_paths:
        # Read data from CSV
        data = pd.read_csv(file_path, header=None)

        # Set up a figure with 4 subplots (one for each feature)
        fig, axs = plt.subplots(1, num_features, figsize=(20, 5), sharey=True)
        fig.suptitle(f'Feature Values Across Indices in {file_path.split("/")[-1]}', fontsize=16)

        # Plot bar graph for each feature
        for i in range(num_features):
            axs[i].bar(range(len(data)), data[i], color='skyblue')
            axs[i].set_title(f'Feature {i+1}')
            axs[i].set_xlabel('Index')
            axs[i].set_ylabel('Feature Value')
            axs[i].grid(True)

        plt.tight_layout()
        plt.show()


def plot_confusion_matrix(name, conf_mat):
    df_cm = pd.DataFrame(conf_mat, index = [i for i in class_name],columns = [i for i in class_name])
    fig = plt.figure(figsize = (10,7))

    fig.suptitle(name, fontsize=16)

    sn.heatmap(df_cm, annot=True)
    plt.savefig(f'{name}.pdf',format="pdf")



