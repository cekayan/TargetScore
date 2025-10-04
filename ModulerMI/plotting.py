"""
plotting.py
Contains plotting functions and related utility functions.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

def plot_tsne_ts(data, matrix, column):
    """
    Plots a t-SNE scatter plot colored by the given column in the data.
    """
    names = set(data[column])
    palette = sns.color_palette("hsv", len(names))
    element_color_dict = {element: palette[i] for i, element in enumerate(names)}
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(matrix)
    
    plt.figure(figsize=(8, 6))
    scatter_fig = plt.gcf()
    scatter_ax = scatter_fig.add_subplot(111)
    
    for i in range(data.shape[0]):
        scatter_ax.scatter(tsne_results[i, 0], tsne_results[i, 1],
                           c=[element_color_dict[list(data[column])[i]]],
                           marker='.')
    
    scatter_ax.set_title(f"t-SNE, {column} colored")
    scatter_ax.set_xlabel("t-SNE Dim. 1")
    scatter_ax.set_ylabel("t-SNE Dim. 2")
    plt.show()

def plot_tsne_input(data, data2, column, pcas):
    """
    Another t-SNE plot function that processes several features.
    """
    names = set(data[column])
    palette = sns.color_palette("hsv", len(names))
    element_color_dict = {element: palette[i] for i, element in enumerate(names)}
    tsne = TSNE(n_components=2, random_state=42)
    
    # For demonstration purposes, assume we have a global drug2target and other dicts.
    # In a full project, these should be passed in or imported from a configuration.
    # Here we simply use the input features from data.
    drug_vecs = np.array([item for item in data['Drug-Name']])
    tsne_results = tsne.fit_transform(drug_vecs.reshape(-1, 1))
    
    plt.figure(figsize=(8, 6))
    scatter_fig = plt.gcf()
    scatter_ax = scatter_fig.add_subplot(111)
    
    for i in range(data.shape[0]):
        scatter_ax.scatter(tsne_results[i, 0], tsne_results[i, 1],
                           c=[element_color_dict[list(data[column])[i]]],
                           marker='.')
    
    scatter_ax.set_title(f"t-SNE, {column} colored")
    scatter_ax.set_xlabel("t-SNE Dim. 1")
    scatter_ax.set_ylabel("t-SNE Dim. 2")
    plt.show()

def classification_metric(predicted, true, bins):
    """
    Computes a classification accuracy by assigning intervals to predicted/true values.
    """
    if predicted.shape[0] != true.shape[0]:
        raise ValueError("Predicted and true arrays must have the same length.")
    
    predicted_classes = np.digitize(predicted, bins) - 1
    true_classes = np.digitize(true, bins) - 1
    num_bins = len(bins) - 1
    predicted_classes[predicted_classes == num_bins] = num_bins - 1
    true_classes[true_classes == num_bins] = num_bins - 1
    accuracy = accuracy_score(true_classes, predicted_classes)
    return accuracy
