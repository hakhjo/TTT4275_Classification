from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import mpl_toolkits.mplot3d
import seaborn as sn
import numpy as np
from scipy.stats import norm, pearsonr
from data import *


plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams["text.usetex"] = True

    

def produce_histograms(x, y):
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
            
            axs[i].hist(feature_values, bins=bins, alpha=0.4, label=class_names[j], density=True, edgecolor='black')
            normal_dist = norm.pdf(x_values, mean, std)
            axs[i].plot(x_values, normal_dist)
        
        for mean in mean_features[:, i]:
            axs[i].axvline(x=mean, color='r', linestyle='dashed', linewidth=1)
        
        axs[i].set_title(feature_names[i], fontsize=25)
        axs[i].tick_params(axis='both', which='major', labelsize=20)  
        if i == 1:
            axs[i].legend(fontsize = 20)    
    plt.tight_layout()
    plt.savefig("features_IRIS.pdf", format="pdf")




def plot_confusion_matrix(name, conf_mat):
    df_cm = pd.DataFrame(conf_mat, index = [i for i in class_names_short],columns = [i for i in class_names_short])

    g = sn.heatmap(df_cm, annot=True, annot_kws={'size':26}, cbar=False, square=True, fmt='g')
    g.set_xticklabels([i for i in class_names_short], fontsize= 18)
    g.set_yticklabels([i for i in class_names_short], fontsize= 18)
    plt.xlabel("$\\hat{\\omega}$", fontsize=20)
    plt.ylabel("$\\omega$", fontsize=20)
    plt.tight_layout()
    plt.savefig(f'{name}.pdf',format="pdf")
    plt.clf()

def plot_correlation_matrix(data):  
    df = pd.DataFrame(data, columns=feature_names)
    
    correlation_matrix = df.corr()
    
    plt.figure(figsize=(8, 6))
    g = sn.heatmap(correlation_matrix, annot=True,annot_kws={'size':26}, fmt=".2f", cmap='coolwarm', 
                cbar=False, linewidths=0.5, linecolor='w')
    g.set_xticklabels([i for i in feature_names], fontsize= 14)
    g.set_yticklabels([i for i in feature_names], fontsize= 14)
    plt.savefig("iris_corr_features.pdf", format="pdf")




def pair_plot(x, y):
    df = pd.DataFrame(x, columns=feature_names)
    t = np.argmax(y, axis=1)
    sn.set_theme(font_scale=2, rc={'text.usetex': True, "mathtext.fontset" : "stix","font.family":"STIXGeneral" })
    g = sn.PairGrid(df)

    for i, j in zip(*np.triu_indices_from(g.axes, k=1)):
        ax = g.axes[i, j]
        r, _ = pearsonr(df.iloc[:, i], df.iloc[:, j])
        ax.annotate(f'$\\rho$ = {r:.2f}', xy=(.5, .5), xycoords='axes fraction', ha='center')
        ax.set_axis_off()

    for i in range(len(feature_names)):
        ax = g.axes[i, i]
        ax.clear()  
        ax.text(0.5, 0.5, feature_names[i], transform=ax.transAxes,
                ha="center", va="center", fontsize=20)
        ax.set_axis_off()

    g.hue_names = class_names
    g.map_lower(sn.scatterplot, hue=[class_names[i] for i in t])
    g.add_legend()
    g.set(xlabel='', ylabel='')
    plt.savefig("scatter_plots.pdf", format="pdf")

