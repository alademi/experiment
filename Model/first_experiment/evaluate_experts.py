import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler
import torch

from Model import util
from Model.util import perform_clustering, save_result_csv
from Model.models_config import ModelBuilder  # PyTorch ModelBuilder

HORIZON = 1
WINDOW_SIZE = 7

MODELS = ModelBuilder.get_available_models()

scaler = StandardScaler()

#############################################
# Evaluate Validation Predictions Point-by-Point
#############################################
def eval_val(model, model_name, subsequences, labels):
    """
    For each validation window (subsequence), adjust its shape as needed and
    obtain a prediction from the model. Then compute RMSE over all windows.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    for t in range(len(subsequences)):
        current_window = subsequences[t]
        if model_name != "mlp":
            # For non-MLP models, ensure the window is 2D (WINDOW_SIZE, HORIZON)
            if current_window.ndim == 1:
                current_window = np.expand_dims(current_window, axis=-1)
        else:
            # For MLP, if shape is (WINDOW_SIZE, 1) squeeze to (WINDOW_SIZE,)
            if current_window.ndim == 2 and current_window.shape[1] == 1:
                current_window = np.squeeze(current_window, axis=1)
        # Add batch dimension
        input_tensor = torch.tensor(current_window, dtype=torch.float32, device=device).unsqueeze(0)
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            # Unpack if model returns a tuple (e.g., DeepAR returns (mean, sigma))
            if isinstance(output, tuple):
                output = output[0]
        predictions.append(output.cpu().numpy().squeeze())
    predictions = np.array(predictions)
    labels = np.array(labels).squeeze()
    mse = np.mean((predictions - labels) ** 2)
    rmse = np.sqrt(mse)
    return rmse

#############################################
# Rank Expert Models Per Cluster
#############################################
def rank_models(dataset, clusters, clusters_no):
    """
    For each cluster, load all expert models (one per model type), evaluate them on
    the cluster's validation set (point-by-point), and choose the one with the lowest RMSE.
    The ranking is saved as a JSON file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_ranking = {"file_name": dataset, "clusters": {}}
    expert_models = []  # Will store tuples: (model_name, model)

    for i in range(clusters_no):
        val_windows = clusters[i]["val_windows"]
        val_labels = clusters[i]["val_labels"]

        model_results = {}
        best_rmse = float("inf")
        best_model_tuple = None

        for model_name in MODELS:
            model_path = (
                f"/Users/aalademi/PycharmProjects/experiment/Model/first_experiment/"
                f"expert/expert-models/{dataset}/cluster{i + 1}/{model_name}"
            )
            checkpoint_file = os.path.join(model_path, "best_model.pth")
            model_builder = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, n_features=HORIZON)
            model = model_builder.build_model().to(device)
            # Load the state dict with weights_only=True to avoid the pickle warning.
            state_dict = torch.load(checkpoint_file, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            rmse_value = eval_val(model, model_name, val_windows, val_labels)
            model_results[model_name] = rmse_value

            if rmse_value < best_rmse:
                best_rmse = rmse_value
                best_model_tuple = (model_name, model)

        # Convert RMSE to native float for JSON serialization.
        sorted_models = sorted(model_results.items(), key=lambda x: x[1])
        ranking_list = [{"model": m, "rmse": float(r)} for m, r in sorted_models]
        models_ranking["clusters"][f"cluster_{i + 1}"] = {"ranking": ranking_list}
        expert_models.append(best_model_tuple)

    json_file_path = f"expert/ranking/{dataset}_ranking.json"
    with open(json_file_path, "w") as json_file:
        json.dump(models_ranking, json_file, indent=4)
    print(f"Ranking results saved to {json_file_path}")
    return expert_models

#############################################
# Cluster Data
#############################################
def cluster_data(train):
    values = train.iloc[:, 1].to_numpy()
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
    return clusters, clustering_result.n_clusters, clustering_result.cluster_centers_

#############################################
# Predict for a Single Test Window
#############################################
def predict(expert_models, clusters_center, test_window):
    """
    For a given test window, determine the closest cluster (using Euclidean distance)
    and then use the corresponding expert model to predict the output.
    The test window is processed point-by-point.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    min_euc_dist = float('inf')
    current_window = test_window.reshape(1, -1)
    closest_cluster_idx = None

    for idx, center in enumerate(clusters_center):
        center = center.reshape(1, -1)
        euclidean_distance = euclidean_distances(current_window, center)[0][0]
        if euclidean_distance < min_euc_dist:
            min_euc_dist = euclidean_distance
            closest_cluster_idx = idx

    print("---------------------------------------------")
    print(f"Closest cluster index: {closest_cluster_idx}")
    # Retrieve the expert model tuple: (model_name, model)
    model_name, expert_model = expert_models[closest_cluster_idx]
    # Prepare test window for prediction based on model type.
    if model_name == "mlp":
        # For MLP, if test_window has shape (WINDOW_SIZE, 1) squeeze to (WINDOW_SIZE,)
        if test_window.ndim == 2 and test_window.shape[1] == 1:
            test_window = np.squeeze(test_window, axis=1)
        # Now input should be 1D of length WINDOW_SIZE; add batch dimension.
        input_tensor = torch.tensor(test_window, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        # For non-MLP, ensure test_window is 2D (WINDOW_SIZE, HORIZON)
        if test_window.ndim == 1:
            test_window = np.expand_dims(test_window, axis=-1)
        input_tensor = torch.tensor(test_window, dtype=torch.float32, device=device).unsqueeze(0)
    expert_model.eval()
    with torch.no_grad():
        output = expert_model(input_tensor)
        # Unpack tuple outputs if necessary.
        if isinstance(output, tuple):
            output = output[0]
    prediction = output.cpu().numpy().squeeze()
    return prediction

#############################################
# Evaluate Expert Models on Test Set (Point-by-Point)
#############################################
def evaluate_expert_models(expert_models, test, clusters_center):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    test = scaler.transform(test.iloc[:, 1].values.reshape(-1, 1))
    test_windows, test_labels = util.make_windows(test, WINDOW_SIZE, HORIZON)

    for t in range(len(test_windows)):
        prediction = predict(expert_models, clusters_center, test_windows[t])
        predictions.append(prediction)

    predictions = np.array(predictions)
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = np.squeeze(predictions, axis=1)
    labels = np.array(test_labels).squeeze()
    mse = np.mean((predictions - labels) ** 2)
    rmse = np.sqrt(mse)
    result = {"rmse": rmse}
    return result

#############################################
# Prepare Data and Run Evaluation
#############################################
def prepare_data(data_path):
    file_names = os.listdir(data_path)
    for name in file_names:
        if name.lower().endswith('.csv'):
            dataset_name = os.path.splitext(name)[0]
            print("Processing dataset:", dataset_name)
            data = pd.read_csv(os.path.join(data_path, name))
            split = int(0.8 * len(data))
            train, test = data[:split], data[split:]
            scaler.fit(train.iloc[:, 1].values.reshape(-1, 1))
            clusters, clusters_no, clusters_center = cluster_data(train)
            expert_models = rank_models(dataset_name, clusters, clusters_no)
            print("Expert models loaded.")
            result = evaluate_expert_models(expert_models, test, clusters_center)
            save_result_csv(dataset_name, result, csv_filename="/Users/aalademi/PycharmProjects/experiment/Model/first_experiment/expert/results/expert_results.csv")

#############################################
# Main
#############################################
test_files_path = "test-models"  # Adjust the path as needed
prepare_data(test_files_path)
