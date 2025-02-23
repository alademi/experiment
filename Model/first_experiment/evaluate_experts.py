import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler
import torch

from Model import util
from Model.util import perform_clustering
from Model.models_config import ModelBuilder

HORIZON = 1
WINDOW_SIZE = 7

MODELS = ModelBuilder.get_available_models()

scaler = StandardScaler()


def eval_val(model, model_name, subsequences, labels):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    for t in range(len(subsequences)):
        current_window = subsequences[t]
        if model_name != "mlp":
            if current_window.ndim == 1:
                current_window = np.expand_dims(current_window, axis=-1)
        else:
            if current_window.ndim == 2 and current_window.shape[1] == 1:
                current_window = np.squeeze(current_window, axis=1)
        # Add batch dimension
        input_tensor = torch.tensor(current_window, dtype=torch.float32, device=device).unsqueeze(0)

        # For torch-based models:
        if hasattr(model, 'eval'):
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                if isinstance(output, tuple):
                    output = output[0]
            pred = output.cpu().numpy().squeeze()
        else:
            # For non-torch models:
            input_array = input_tensor.cpu().numpy().squeeze()
            if input_array.ndim == 1:
                input_array = input_array.reshape(1, -1)
            pred = model.predict(input_array).squeeze()

        # If pred is an array (e.g., [value1, value2, value3]), take the first element.
        if isinstance(pred, np.ndarray) and pred.ndim != 0:
            pred = pred[0]
        predictions.append(pred)

    predictions = np.array(predictions)
    labels = np.array(labels).squeeze()
    mse = np.mean((predictions - labels) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def rank_models(dataset, clusters, clusters_no):
    """
    For each cluster, load all expert models (one per model type), evaluate them on
    the cluster's validation set (point-by-point), and choose the one with the lowest RMSE.
    The ranking is saved as a JSON file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_ranking = {"file_name": dataset, "clusters": {}}
    expert_models = []

    for i in range(clusters_no):
        val_windows = clusters[i]["val_windows"]
        val_labels = clusters[i]["val_labels"]

        model_results = {}
        best_rmse = float("inf")
        best_model_tuple = None
        import joblib  # Make sure to import joblib

        for model_name in MODELS:
            model_path = (
                f"/Users/aalademi/PycharmProjects/experiment/Model/first_experiment/"
                f"expert/expert-models/{dataset}/cluster{i + 1}/{model_name}"
            )

            # For non-torch models:
            if model_name in ["decision_tree", "random_forest", "xgboost"]:
                checkpoint_file = os.path.join(model_path, "best_model.pkl")
                model = joblib.load(checkpoint_file)
            else:
                checkpoint_file = os.path.join(model_path, "best_model.pth")
                model_builder = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, horizon=HORIZON)
                model = model_builder.build_model().to(device)
                state_dict = torch.load(checkpoint_file, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)

            # Then evaluate the model...
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

    json_file_path = f"results/experts_ranking"

    if not os.path.exists(json_file_path):
        print(f"Directory {json_file_path} does not exist. Creating directory...")
        os.makedirs(json_file_path)

    json_file_name = f"expert/ranking/{dataset}_ranking.json"
    with open(json_file_name, "w") as json_file:
        json.dump(models_ranking, json_file, indent=4)
    print(f"Ranking results saved to {json_file_name}")
    return expert_models


def cluster_data(train):
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


def predict(expert_models, clusters_center, test_window):
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
    model_name, expert_model = expert_models[closest_cluster_idx]

    # Branch based on model type.
    if model_name in ["decision_tree", "random_forest", "xgboost"]:
        # For non-torch models, ensure the input shape matches training expectations.
        # Adjust shape if necessary. For example, if training used 2D arrays:
        if test_window.ndim != 2:
            test_window = test_window.reshape(1, -1)
        prediction = expert_model.predict(test_window)
        prediction = np.array(prediction).squeeze()
    else:
        # For torch-based models, adjust the input as before.
        if model_name == "mlp":
            if test_window.ndim == 2 and test_window.shape[1] == 1:
                test_window = np.squeeze(test_window, axis=1)
            input_tensor = torch.tensor(test_window, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            if test_window.ndim == 1:
                test_window = np.expand_dims(test_window, axis=-1)
            input_tensor = torch.tensor(test_window, dtype=torch.float32, device=device).unsqueeze(0)
        expert_model.eval()
        with torch.no_grad():
            output = expert_model(input_tensor)
            if isinstance(output, tuple):
                output = output[0]
        prediction = output.cpu().numpy().squeeze()
    return prediction


def evaluate_expert_models(expert_models, test, clusters_center, training_range):
    """
    Evaluates expert models on the test set.
    Predictions are generated on scaled windows and then inverse-transformed.
    The RMSE is computed on the original scale and then normalized using
    the training data's range.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []

    # Scale test values for window creation
    test_scaled = scaler.transform(test.iloc[:, 1].values.reshape(-1, 1))
    test_windows, test_labels = util.make_windows(test_scaled, WINDOW_SIZE, HORIZON)

    for t in range(len(test_windows)):
        prediction = predict(expert_models, clusters_center, test_windows[t])
        predictions.append(prediction)

    predictions = np.array(predictions)
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = np.squeeze(predictions, axis=1)
    labels = np.array(test_labels).squeeze()

    # Inverse-transform predictions and labels back to the original scale
    predictions_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).squeeze()
    labels_orig = scaler.inverse_transform(labels.reshape(-1, 1)).squeeze()

    mse_orig = np.mean((predictions_orig - labels_orig) ** 2)
    rmse_orig = np.sqrt(mse_orig)

    result = {"expert-models": rmse_orig}
    return result


def save_result_csv(dataset_name, result, csv_dir):
    """
    Saves (or appends) the result for a given dataset to a CSV file.
    If result is not a dictionary, it will be stored under the key 'result'.
    """
    csv_filename = f"{csv_dir}/expert_results.csv"

    if not os.path.exists(csv_dir):
        print(f"Directory {csv_dir} does not exist. Creating directory...")
        os.makedirs(csv_dir)

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

            values = data.iloc[:, 1].values
            # If there are more than 20,000 rows, select only the first 20,000 and log the confirmation
            if len(values) > 20000:
                data = data.iloc[:20000]
                print(f"Dataset {dataset_name} has more than 20000 rows. Selected the first 20000 rows for processing.")

            print(f"Data shape for {dataset_name} after processing: {data.shape}")

            split = int(0.8 * len(data))
            train, test = data[:split], data[split:]

            # Fit the scaler on training data
            scaler.fit(train.iloc[:, 1].values.reshape(-1, 1))
            # Compute the training range from the original (unscaled) training values
            train_min = train.iloc[:, 1].min()
            train_max = train.iloc[:, 1].max()
            training_range = train_max - train_min

            clusters, clusters_no, clusters_center = cluster_data(train)
            expert_models = rank_models(dataset_name, clusters, clusters_no)
            print("Expert models loaded.")
            result = evaluate_expert_models(expert_models, test, clusters_center, training_range)
            save_result_csv(dataset_name, result,
                            csv_dir="/Users/aalademi/PycharmProjects/experiment/Model/first_experiment/results/evaluation")


test_files_path = "test"  # Adjust the path as needed
prepare_data(test_files_path)
