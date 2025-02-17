import os

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from tslearn.clustering import KShape

MODELS = ["conv", "conv-lstm", "lstm" ,"mlp", "ar"]


def get_models_list() :
    return MODELS

def evaluate_preds(y_true, y_pred, scaler):
    # Ensure y_pred is a NumPy array in case it's a list of predictions
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    # Optionally ensure y_true is a NumPy array
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)

    # Convert to tensors for metric calculations
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)

    # Rescale the predictions and ground truth to the original scale
  #  y_true_rescaled = scaler.inverse_transform(y_true.numpy().reshape(-1, 1)).reshape(-1)
   # y_pred_rescaled = scaler.inverse_transform(y_pred.numpy().reshape(-1, 1)).reshape(-1)

    # Calculate various metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
    rmse = tf.sqrt(mse)
  #  rmse_norm = np.sqrt(mean_squared_error(y_true.numpy(), y_pred.numpy()))
#    r2 = r2_score(y_true, y_pred)

    return {
        "mae": mae.numpy(),
        "mse": mse.numpy(),
        "rmse": rmse.numpy(),
       # "rmse_norm": rmse_norm,
    #    "r2": r2
    }


def compute_clusters_kshape(subsequences):
    # Ensure the data is 2D: (n_samples, time_steps)
    if subsequences.ndim > 2:
        n_samples = subsequences.shape[0]
        subsequences = subsequences.reshape(n_samples, -1)

    # Evaluate a range of clusters to determine the optimal number via silhouette score.
    # Note: silhouette_score here uses Euclidean distance. Although not the exact shape-based distance,
    # it can serve as a proxy since the subsequences are already z-normalized.
    range_n_clusters = range(2, 11)
    silhouette_scores = []

    for n_clusters in range_n_clusters:
        kshape = KShape(n_clusters=n_clusters, random_state=0)
        labels = kshape.fit_predict(subsequences)
        score = silhouette_score(subsequences, labels, metric="euclidean")
        silhouette_scores.append(score)

    optimal_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_n_clusters}")
    return optimal_n_clusters


def perform_clustering(subsequences):
    # Ensure the data is 2D: (n_samples, time_steps)
    if subsequences.ndim > 2:
        n_samples = subsequences.shape[0]
        subsequences = subsequences.reshape(n_samples, -1)

    clusters_no = compute_clusters_kshape(subsequences)
    kshape = KShape(n_clusters=clusters_no, random_state=42)
    kshape.fit(subsequences)
    return kshape

def get_labelled_windows(x, horizon=1):
    return x[:, :-horizon], x[:, -horizon:]

def make_windows(x, window_size=7, horizon=1):
    n_samples = x.shape[0]
    total_length = window_size + horizon
    n_windows = n_samples - total_length + 1
    window_indexes = np.arange(total_length) + np.arange(n_windows)[:, None]
    windowed_array = x[window_indexes]
    if x.shape[1] == 1:
        windowed_array = windowed_array.squeeze(-1)
    windows, labels = get_labelled_windows(windowed_array, horizon=horizon)
    return windows, labels


def create_model_checkpoint(save_path):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=save_path,
        verbose=0,
        save_best_only=True
    )

def save_sklearn_model(model, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(model, save_path)


def save_result_csv(dataset_name, result, csv_filename="results/expert_results.csv"):
    """
    Saves (or appends) the result for a given dataset to a CSV file.
    If result is not a dictionary, it will be stored under the key 'result'.
    """
    # Build a dictionary for the row to save.
    if isinstance(result, dict):
        data = {"dataset": dataset_name, **result}
    else:
        data = {"dataset": dataset_name, "result": result}
    df = pd.DataFrame([data])

    # If the CSV file doesn't exist, write with header; otherwise, append without header.
    if not os.path.exists(csv_filename):
        df.to_csv(csv_filename, index=False)
    else:
        df.to_csv(csv_filename, index=False, mode='a', header=False)
    print(f"Results for {dataset_name} saved to {csv_filename}")