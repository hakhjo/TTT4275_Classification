
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
from scipy.stats import norm

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




def produce_histograms(x, t):
    class_labels = np.argmax(t, axis=1)
    # Number of classes and features
    num_classes = t.shape[1]
    num_features = x.shape[1]
    
    mean_features = np.zeros((num_classes, num_features))
    
    # Calculate mean features for each class
    for i in range(num_classes):
        mean_features[i, :] = np.mean(x[class_labels == i], axis=0)
    
    print(mean_features)
    
    fig, axs = plt.subplots(num_features, 1, figsize=(10, 8))
    
    for i in range(num_features):
        feature_min = np.min(x[:, i])
        feature_max = np.max(x[:, i])
        bins = np.arange(feature_min, feature_max + 0.1, 0.1)  # creating bins of width 0.1
        x_values = np.linspace(feature_min, feature_max, 300)
        for j in range(num_classes):
            # Prepare data for histogram: extract all feature values for class 'j'
            feature_values = x[class_labels == j, i]
            mean = np.mean(feature_values)
            std = np.std(feature_values)
            print(mean, std)
            
            axs[i].hist(feature_values, bins=bins, alpha=0.4, label=f'Class {j+1} ', density=True, edgecolor='black')
            normal_dist = norm.pdf(x_values, mean, std)
            axs[i].plot(x_values, normal_dist, label=f'Class {j+1} PDF')
        
        for mean in mean_features[:, i]:
            axs[i].axvline(x=mean, color='r', linestyle='dashed', linewidth=1)
        
        axs[i].set_title(f'Feature {i+1}:')
        axs[i].legend()
    
    plt.tight_layout()
    plt.show()


def produce_histograms1(x,t):
   # Decode the one-hot encoded labels to get the class indices
    class_labels = np.argmax(t, axis=1)
    
    # Number of classes and features
    num_classes = t.shape[1]
    num_features = x.shape[1]
    
    # Plotting histograms
    fig, axs = plt.subplots(num_features, 1, figsize=(10, 8))
    
    if num_features == 1:
        axs = [axs]  # make it iterable
    
    for i in range(num_features):
        # Define bins for this feature across all classes
        feature_min = np.min(x[:, i])
        feature_max = np.max(x[:, i])
        bins = np.arange(feature_min, feature_max + 0.1, 0.1)  # creating bins of width 0.1
        
        # Calculate the x points for the Gaussian curve
        x_values = np.linspace(feature_min, feature_max, 300)
        
        for j in range(num_classes):
            # Prepare data for histogram: extract all feature values for class 'j'
            feature_values = x[class_labels == j, i]
            mean = np.mean(feature_values)
            std = np.std(feature_values)
            
            # Plot histogram for feature 'i' of class 'j'
            axs[i].hist(feature_values, bins=bins, alpha=0.4, label=f'Class {j+1}', density=True, edgecolor='black')
            
            # Plot Gaussian distribution for the current class and feature
            normal_dist = norm.pdf(x_values, mean, std)
            axs[i].plot(x_values, normal_dist, label=f'Class {j+1} fit')

        # Plot vertical lines for the mean values of each class for the current feature
        for mean in np.mean(x[class_labels == j], axis=0):
            axs[i].axvline(x=mean, color='r', linestyle='dashed', linewidth=1)
        
        axs[i].set_title(f'Feature {i+1}: Distribution with Class Means and Gaussian Fit')
        axs[i].legend()

    plt.tight_layout()
    plt.show()