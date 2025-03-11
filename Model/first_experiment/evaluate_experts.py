import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler
import torch
import joblib

from Model import util
from Model.util import perform_clustering
from Model.models_config import ModelBuilder

HORIZON = 1
WINDOW_SIZE = 7
MODELS = ModelBuilder.get_available_models()
scaler = StandardScaler()


def eval_val(model, model_name, subsequences, labels):
    """
    Evaluate a model on each subsequence individually and return the RMSE.
    """
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
        # Add batch dimension.
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

        # If pred is an array, take the first element.
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

        for model_name in MODELS:
            model_path = (
                f"/Users/aalademi/PycharmProjects/first_experiment/Model/first_experiment/"
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

            # Evaluate the model on the validation windows.
            rmse_value = eval_val(model, model_name, val_windows, val_labels)
            model_results[model_name] = rmse_value

            if rmse_value < best_rmse:
                best_rmse = rmse_value
                best_model_tuple = (model_name, model)

        # Save ranking information for the cluster.
        sorted_models = sorted(model_results.items(), key=lambda x: x[1])
        ranking_list = [{"model": m, "rmse": float(r)} for m, r in sorted_models]
        models_ranking["clusters"][f"cluster_{i + 1}"] = {"ranking": ranking_list}
        expert_models.append(best_model_tuple)

    json_file_path = "results/experts_ranking"
    if not os.path.exists(json_file_path):
        print(f"Directory {json_file_path} does not exist. Creating directory...")
        os.makedirs(json_file_path)

    json_file_name = f"results/experts_ranking/{dataset}_ranking.json"
    with open(json_file_name, "w") as json_file:
        json.dump(models_ranking, json_file, indent=4)
    print(f"Ranking results saved to {json_file_name}")
    return expert_models


def cluster_data(train):
    """
    Cluster the training windows and split them into training and validation subsets.
    """
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
    """
    For a given test window, identify the closest cluster center and use the corresponding
    expert model to generate a prediction. Returns both the prediction and the cluster index.
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
    model_name, expert_model = expert_models[closest_cluster_idx]

    # Branch based on model type.
    if model_name in ["decision_tree", "random_forest", "xgboost"]:
        if test_window.ndim != 2:
            test_window = test_window.reshape(1, -1)
        prediction = expert_model.predict(test_window)
        prediction = np.array(prediction).squeeze()
    else:
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

    return prediction, closest_cluster_idx + 1


def evaluate_expert_models_and_save_predictions(expert_models, test, clusters_center, dataset_name):
    """
    Evaluates expert models on the test set. Predictions are generated on scaled windows and then
    inverse-transformed. Saves a CSV file with actual values, expert model predictions, and cluster index.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = []
    cluster_indices = []

    # Scale test values for window creation.
    test_scaled = scaler.transform(test.iloc[:, 1].values.reshape(-1, 1))
    test_windows, test_labels = util.make_windows(test_scaled, WINDOW_SIZE, HORIZON)

    for t in range(len(test_windows)):
        prediction, cluster_idx = predict(expert_models, clusters_center, test_windows[t])
        predictions.append(prediction)
        cluster_indices.append(cluster_idx)

    predictions = np.array(predictions)
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = np.squeeze(predictions, axis=1)
    labels = np.array(test_labels).squeeze()

    # Inverse-transform predictions and labels back to the original scale.
    predictions_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).squeeze()
    labels_orig = scaler.inverse_transform(labels.reshape(-1, 1)).squeeze()

    mse_orig = np.mean((predictions_orig - labels_orig) ** 2)
    rmse_orig = np.sqrt(mse_orig)
    result = {"expert-models": rmse_orig}
    print(f"Expert-model RMSE on original scale: {rmse_orig}")

    predictions_file = f"/Model/first_experiment/results/predictions_upd/{dataset_name}_predictions.csv"

    # Update or create the predictions file.
    if os.path.exists(predictions_file):
        existing_df = pd.read_csv(predictions_file)
        existing_df["expert-models"] = predictions_orig
        existing_df["Cluster"] = cluster_indices
        existing_df.to_csv(predictions_file, index=False)
        print(f"Expert predictions and cluster indices updated in existing file {predictions_file}")
    else:
        predictions_df = pd.DataFrame({
            "Actual": labels_orig,
            "expert-models": predictions_orig,
            "Cluster": cluster_indices
        })
        predictions_df.to_csv(predictions_file, index=False)
        print(f"Expert predictions and cluster indices saved to new file {predictions_file}")

    return result


def save_result_csv(dataset_name, result, csv_dir):
    """
    Saves (or updates) the evaluation result for a given dataset to a CSV file.
    If a row for the dataset already exists, its columns are updated; otherwise, a new row is appended.
    """
    csv_filename = f"{csv_dir}/results3.csv"
    # Create directory if it doesn't exist.
    if not os.path.exists(csv_dir):
        print(f"Directory {csv_dir} does not exist. Creating directory...")
        os.makedirs(csv_dir)

    # Prepare the new data as a DataFrame.
    new_data = pd.DataFrame([{"Dataset": dataset_name, **result}])

    if os.path.exists(csv_filename):
        df = pd.read_csv(csv_filename)
        # Check if a row for the dataset already exists.
        if dataset_name in df["Dataset"].values:
            df.loc[df["Dataset"] == dataset_name, list(result.keys())] = list(result.values())
            print(f"Updated evaluation result for {dataset_name} in {csv_filename}")
        else:
            df = pd.concat([df, new_data], ignore_index=True)
            print(f"Appended evaluation result for {dataset_name} to {csv_filename}")
        df.to_csv(csv_filename, index=False)
    else:
        new_data.to_csv(csv_filename, index=False)
        print(f"Created new evaluation results file {csv_filename}")



def prepare_data(data_path):
    file_names = os.listdir(data_path)
    for name in file_names:
        if name.lower().endswith('.csv'):
            dataset_name = os.path.splitext(name)[0]
            print("Processing dataset:", dataset_name)
            data = pd.read_csv(os.path.join(data_path, name))

            values = data.iloc[:, 1].values
            if len(values) > 20000:
                data = data.iloc[:20000]
                print(f"Dataset {dataset_name} has more than 20000 rows. Selected the first 20000 rows for processing.")
            print(f"Data shape for {dataset_name} after processing: {data.shape}")

            split = int(0.8 * len(data))
            train, test = data[:split], data[split:]

            # Fit the scaler on training data.
            scaler.fit(train.iloc[:, 1].values.reshape(-1, 1))

            clusters, clusters_no, clusters_center = cluster_data(train)
            expert_models = rank_models(dataset_name, clusters, clusters_no)
            print("Expert models loaded.")
            result = evaluate_expert_models_and_save_predictions(expert_models, test, clusters_center,
                                                                 dataset_name)
            save_result_csv(dataset_name, result,
                            csv_dir="/Model/first_experiment/results/evaluation")


test_files_path = "test"
prepare_data(test_files_path)
