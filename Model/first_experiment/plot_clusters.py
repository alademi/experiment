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


def plot_clusters(subsequences, labels, cluster_no, window_size, file_name):
    epsilon = 1e-8
    subsequences_norm = (subsequences - np.mean(subsequences, axis=1, keepdims=True)) / (
                np.std(subsequences, axis=1, keepdims=True) + epsilon)

    # Define a list of distinct colors for the clusters (will repeat if clusters > len(cluster_colors))
    cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'cyan', 'magenta', 'gray']

    # Dynamically determine subplot layout
    rows = int(np.ceil(cluster_no / 3))
    plt.figure(figsize=(18, 12))  # Adjust figure size for clarity

    for cluster in range(cluster_no):
        plt.subplot(rows, 3, cluster + 1)
        # Get standardized series for the current cluster
        cluster_series = subsequences_norm[labels == cluster]

        # Plot each standardized time series in the cluster, flattening in case of extra dimensions
        for series in cluster_series:
            plt.plot(range(1, window_size + 1), series.flatten(),
                     color=cluster_colors[cluster % len(cluster_colors)], alpha=0.5)

        # Set fixed x-axis limits based on window size
        plt.xlim(1, window_size)
        # Set consistent y-axis limits for z-scores
        plt.ylim(-3, 3)  # Common z-score range

        # Add grid lines with light style
        plt.grid(True, linestyle='--', alpha=0.5, color='gray')

        # Add titles and labels for clarity
        plt.title(f'Cluster {cluster + 1}', fontsize=16, fontweight='bold', color='black')
        plt.xlabel("Time Index", fontsize=14, color='black')
        plt.ylabel("Value (Z-Score)", fontsize=14, color='black')

    # Save the plot as an image file
    plt.tight_layout()
    save_path = f"cluster_plots/{file_name}_clusters.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Create directory if it doesn't exist
    plt.savefig(save_path)
    plt.close()  # Free memory
    print(f"Saved plot for {file_name} to {save_path}")



def cluster_data(train):
    values = train.iloc[:, 1].to_numpy()
    time_stamp = train.iloc[:, 0].to_numpy()
    #train_norm = scaler.transform(train.iloc[:, 1].values.reshape(-1, 1))
    train_windows, train_labels = util.make_windows(train.iloc[:, 1].values.reshape(-1, 1), WINDOW_SIZE, HORIZON)
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

            values = data.iloc[:, 1].values
            # If there are more than 20,000 rows, select only the first 20,000 and log the confirmation
            if len(values) > 20000:
                data = data.iloc[:20000]
                print(f"Dataset {dataset_name} has more than 20000 rows. Selected the first 20000 rows for processing.")

            # Log the shape of the data to check the truncation
            print(f"Data shape for {dataset_name} after processing: {data.shape}")

            split = int(0.8 * len(data))
            train, test = data[:split], data[split:]
            # Fit the scaler on the training data
            values = train.iloc[:, 1].values
            scaler.fit(values.reshape(-1, 1))
            clusters, train_windows = cluster_data(train)
            plot_clusters(train_windows, clusters.labels_, clusters.n_clusters, WINDOW_SIZE, dataset_name)


# Specify the path to your directory containing CSV files
test_files_path = "test"  # Adjust the path as needed
prepare_data(test_files_path)
