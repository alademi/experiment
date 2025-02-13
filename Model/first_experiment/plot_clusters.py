import os

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from Model import util
from Model.util import perform_clustering

HORIZON = 1
WINDOW_SIZE = 7
scaler = StandardScaler()




def plot_clusters(validation_windows, labels, cluster_no, window_size, file_name):
    """
    Plot each cluster of time series with red, blue, green, and other distinct colors on a white background,
    and save each plot with the file name.

    Parameters:
    - validation_windows: 2D numpy array where each row is a flattened time series.
    - labels: array of cluster labels for each time series.
    - cluster_no: int, the number of clusters.
    - window_size: int, the number of time steps (x-axis size).
    - file_name: str, base name of the file for saving plots.
    """
    # Standardize each time series to have mean 0 and standard deviation 1 (z-scores)
    validation_windows_z = (validation_windows - np.mean(validation_windows, axis=1, keepdims=True)) / np.std(
        validation_windows, axis=1, keepdims=True)

    # Define a list of distinct colors for the clusters
    cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan', 'magenta', 'gray']

    plt.figure(figsize=(18, 12))  # Adjust figure size for clarity

    for cluster in range(cluster_no):
        plt.subplot(cluster_no // 3 + 1, 3, cluster + 1)
        cluster_data = validation_windows_z[labels == cluster]  # Use standardized data

        # Plot each standardized time series in the cluster with distinct colors
        for series in cluster_data:
            plt.plot(range(1, window_size + 1), series, color=cluster_colors[cluster % len(cluster_colors)], alpha=0.5)

        # Set fixed x-axis limits based on window size
        plt.xlim(1, window_size)

        # Set consistent y-axis limits for z-scores
        plt.ylim(-3, 3)  # Common z-score range

        # Add grid lines with light style for contrast on white background
        plt.grid(True, linestyle='--', alpha=0.5, color='gray')

        # Styling for title and labels
        plt.title(f'Cluster {cluster + 1}', fontsize=16, fontweight='bold', color='black')
        plt.xlabel("Time Index", fontsize=14, color='black')
        plt.ylabel("Value (Z-Score)", fontsize=14, color='black')

    # Save plot as an image file
    plt.tight_layout()
    save_path = f"cluster_plots/{file_name}_clusters.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(save_path)
    plt.close()  # Close the plot to free memory
    print(f"Saved plot for {file_name} to {save_path}")


def cluster_data(train):
    values = train.iloc[:, 1].to_numpy()
    time_stamp = train.iloc[:, 0].to_numpy()
    train_norm = scaler.transform(train.iloc[:, 1].values.reshape(-1, 1))
    train_windows, train_labels = util.make_windows(train_norm, WINDOW_SIZE, HORIZON)
    clustering_result = perform_clustering(train_windows)

    clusters = {}
    for i in range(clustering_result.n_clusters):
        cluster_subsequences = train_windows[clustering_result.labels_ == i]
        cluster_labels = train_labels[clustering_result.labels_ == i]
        n_samples = len(cluster_subsequences)
        split_index = int(n_samples * 0.8)
        clusters[i] = {
            "train_windows": cluster_subsequences[:split_index],
            "train_labels": cluster_labels[:split_index],
            "val_windows": cluster_subsequences[split_index:],
            "val_labels": cluster_labels[split_index:]
        }
    return clustering_result, train_windows


def prepare_data(data_path):
    # List all files in the directory
    file_names = os.listdir(data_path)
    for name in file_names:
        # Process only CSV files
        if name.lower().endswith('.csv'):
            # Extract dataset name without the .csv extension
            dataset_name = os.path.splitext(name)[0]
            print("Processing dataset:", dataset_name)
            # Use os.path.join to build the full file path
            data = pd.read_csv(os.path.join(data_path, name))
            split = int(0.8 * len(data))
            train, test = data[:split], data[split:]
            # Fit the scaler on the training data
            values = train.iloc[:, 1].values
            scaler.fit(values.reshape(-1, 1))
            clusters, train_windows = cluster_data(train)
            plot_clusters(train_windows, clusters.labels_, clusters.n_clusters, WINDOW_SIZE, dataset_name)


# Specify the path to your directory containing CSV files
test_files_path = "test_data"  # Adjust the path as needed
prepare_data(test_files_path)
