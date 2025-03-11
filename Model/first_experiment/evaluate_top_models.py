import os
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

        input_tensor = torch.tensor(current_window, dtype=torch.float32, device=device).unsqueeze(0)

        if hasattr(model, 'eval'):
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                if isinstance(output, tuple):
                    output = output[0]
            pred = output.cpu().numpy().squeeze()
        else:
            input_array = input_tensor.cpu().numpy().squeeze()
            if input_array.ndim == 1:
                input_array = input_array.reshape(1, -1)
            pred = model.predict(input_array).squeeze()

        predictions.append(pred)

    predictions = np.array(predictions)
    labels = np.array(labels).squeeze()
    rmse = np.sqrt(np.mean((predictions - labels) ** 2))
    return rmse


def rank_models(dataset, clusters, clusters_no):
    """
    Load expert models per cluster, evaluate them, and store the top 3 models.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    expert_models = {}

    for i in range(clusters_no):
        val_windows = clusters[i]["val_windows"]
        val_labels = clusters[i]["val_labels"]

        model_results = {}
        models = {}

        for model_name in MODELS:
            model_path = f"/Users/aalademi/PycharmProjects/experiment/Model/first_experiment/expert/expert-models/{dataset}/cluster{i + 1}/{model_name}"

            if model_name in ["decision_tree", "random_forest", "xgboost"]:
                checkpoint_file = os.path.join(model_path, "best_model.pkl")
                model = joblib.load(checkpoint_file)
            else:
                checkpoint_file = os.path.join(model_path, "best_model.pth")
                model_builder = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, horizon=HORIZON)
                model = model_builder.build_model().to(device)
                state_dict = torch.load(checkpoint_file, map_location=device, weights_only=True)
                model.load_state_dict(state_dict)

            rmse_value = eval_val(model, model_name, val_windows, val_labels)
            model_results[model_name] = rmse_value
            models[model_name] = model

        sorted_models = sorted(model_results.items(), key=lambda x: x[1])[:3]
        expert_models[i] = {"models": {m: models[m] for m, _ in sorted_models}}

    return expert_models


def get_closest_clusters(test_window, clusters_center, top_k=3):
    """
    Returns the indices of the closest `top_k` clusters.
    """
    distances = [euclidean_distances(test_window.reshape(1, -1), c.reshape(1, -1))[0][0]
                 for c in clusters_center]
    return np.argsort(distances)[:top_k]  # Return indices of the closest clusters


def predict_top_3_models(expert_models, clusters_center, test_window):
    """
    Predict using the top 3 models of the closest cluster and average the results.
    """
    closest_cluster_idx = get_closest_clusters(test_window, clusters_center, top_k=1)[0]
    top_models = expert_models[closest_cluster_idx]["models"]

    predictions = [get_model_prediction(model_name, model, test_window) for model_name, model in top_models.items()]
    return np.mean(predictions)


def predict_closest_3_clusters(expert_models, clusters_center, test_window):
    """
    Predict using the best model from the 3 closest clusters and average their predictions.
    """
    closest_clusters = get_closest_clusters(test_window, clusters_center, top_k=3)
    predictions = []

    for cluster_idx in closest_clusters:
        best_model_name, best_model = next(iter(expert_models[cluster_idx]["models"].items()))  # Get best model
        predictions.append(get_model_prediction(best_model_name, best_model, test_window))

    return np.mean(predictions)


def get_model_prediction(model_name, model, test_window):
    """
    Get the prediction from a single model.
    Ensures consistency in handling different model types.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Handling traditional ML models (decision tree, random forest, XGBoost)
    if model_name in ["decision_tree", "random_forest", "xgboost"]:
        if test_window.ndim != 2:
            test_window = test_window.reshape(1, -1)
        prediction = model.predict(test_window)
        return np.array(prediction).squeeze()

    # Handling neural network models
    if model_name == "mlp":
        if test_window.ndim == 2 and test_window.shape[1] == 1:
            test_window = np.squeeze(test_window, axis=1)

    else:
        if test_window.ndim == 1:
            test_window = np.expand_dims(test_window, axis=-1)

    input_tensor = torch.tensor(test_window, dtype=torch.float32, device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        if isinstance(output, tuple):
            output = output[0]

    return output.cpu().numpy().squeeze()



def predict_top_model(expert_models, clusters_center, test_window):
    """
    Predict using only the best model from the closest cluster.
    """
    # Identify the closest cluster
    closest_cluster_idx = get_closest_clusters(test_window, clusters_center, top_k=1)[0]

    # Retrieve the best model from the closest cluster
    best_model_name, best_model = next(iter(expert_models[closest_cluster_idx]["models"].items()))  # Get best model

    # Generate a prediction using the best model
    prediction = get_model_prediction(best_model_name, best_model, test_window)

    return prediction


def evaluate_models(dataset, expert_models, test, clusters_center):
    """
    Evaluate expert models using two methods:
    - Top-3-Experts: Uses the top 3 models from the closest cluster.
    - Closest-3-Clusters: Uses the best model from the 3 closest clusters.
    """
    test_scaled = scaler.transform(test.iloc[:, 1].values.reshape(-1, 1))
    test_windows, test_labels = util.make_windows(test_scaled, WINDOW_SIZE, HORIZON)

    predictions_top_model = [predict_top_model(expert_models, clusters_center, w) for w in test_windows]
    predictions_top_3 = [predict_top_3_models(expert_models, clusters_center, w) for w in test_windows]
    predictions_closest_3 = [predict_closest_3_clusters(expert_models, clusters_center, w) for w in test_windows]

    rmse_top = np.sqrt(np.mean((scaler.inverse_transform(np.array(predictions_top_model).reshape(-1, 1)).squeeze() -
                                scaler.inverse_transform(np.array(test_labels).reshape(-1, 1)).squeeze()) ** 2))

    # Inverse-transform RMSE back to the original scale
    rmse_top_3 = np.sqrt(np.mean((scaler.inverse_transform(np.array(predictions_top_3).reshape(-1, 1)).squeeze() -
                                  scaler.inverse_transform(np.array(test_labels).reshape(-1, 1)).squeeze()) ** 2))

    rmse_closest_3 = np.sqrt(
        np.mean((scaler.inverse_transform(np.array(predictions_closest_3).reshape(-1, 1)).squeeze() -
                 scaler.inverse_transform(np.array(test_labels).reshape(-1, 1)).squeeze()) ** 2))

    save_average_rmse_csv(dataset, rmse_top, rmse_top_3, rmse_closest_3)


def save_average_rmse_csv(dataset, rmse_top, rmse_top_3, rmse_closest_3):
    """
    Update or create results.csv with new RMSE values for Top-3-Experts and Closest-3-Clusters.
    """
    csv_path = "results/evaluation/results3.csv"
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    new_data = pd.DataFrame([{"Dataset": dataset, "expert-model": rmse_top, "expert-3-models": rmse_top_3, "closest-3-clusters": rmse_closest_3}])

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        if "Dataset" not in df.columns:
            df["Dataset"] = ""

        if dataset in df["Dataset"].values:
            # Update existing row for this dataset
            df.loc[df["Dataset"] == dataset, ["expert-model", "expert-3-models", "closest-3-clusters"]] = [rmse_top,
                                                                                                      rmse_top_3,
                                                                                                      rmse_closest_3]
            print(f"Updated results for {dataset} in {csv_path}")
        else:
            # Append new dataset results
            df = pd.concat([df, new_data], ignore_index=True)
            print(f"Appended new results for {dataset} in {csv_path}")

        df.to_csv(csv_path, index=False)
    else:
        # Create new file if not exists
        new_data.to_csv(csv_path, index=False)
        print(f"Created results file and saved results for {dataset} in {csv_path}")


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
            evaluate_models(dataset_name, expert_models, test, clusters_center)


test_files_path = "test"
prepare_data(test_files_path)
