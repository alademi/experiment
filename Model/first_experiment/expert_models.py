import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import euclidean_distances
from tensorflow.keras import models
from sklearn.preprocessing import StandardScaler

from Model import util
from Model.util import perform_clustering, save_result_csv

HORIZON = 1
WINDOW_SIZE = 7
MODELS = ["deepar"]
scaler = StandardScaler()


def eval_val(model, subsequences, labels):
    predictions = []
    for t in range(len(subsequences)):
        current_window = subsequences[t].reshape(1, -1)
        prediction = model.predict(current_window)
        predictions.append(prediction)

    # Convert list to numpy array; assumed shape is (n, 1, 1)
    predictions = np.array(predictions)

    # Squeeze out the last dimension so that predictions become (n, 1)
    predictions = np.squeeze(predictions, axis=2)

    # Alternatively, if you prefer, you can use reshape:
    # predictions = predictions.reshape(predictions.shape[0], predictions.shape[1])

    labels = np.array(labels)  # assumed shape is (n, 1)

    mse = np.mean((predictions - labels) ** 2)
    rmse = np.sqrt(mse)

    return rmse

def rank_models(dataset, clusters, clusters_no):
    models_ranking = {"file_name": dataset, "clusters": {}}
    expert_models = []

    for i in range(clusters_no):
        val_windows = clusters[i]["val_windows"]
        val_labels = clusters[i]["val_labels"]

        model_results = {}
        best_rmse = float("inf")
        best_model = None

        for model_name in MODELS:
            model_path = (
                f"/Users/aalademi/PycharmProjects/experiment/Model/first_experiment/"
                f"expert/expert-models/{dataset}/cluster{i + 1}/{model_name}"
            )
            model = models.load_model(model_path)
            rmse_value = eval_val(model, val_windows, val_labels)

            model_results[model_name] = rmse_value

            if rmse_value < best_rmse:
                best_rmse = rmse_value
                best_model = model

        sorted_models = sorted(model_results.items(), key=lambda x: x[1])
        ranking_list = [{"model": m, "rmse": r} for m, r in sorted_models]
        models_ranking["clusters"][f"cluster_{i + 1}"] = {"ranking": ranking_list}
        expert_models.append(best_model)

    json_file_path = f"expert/ranking/{dataset}_ranking.json"
    with open(json_file_path, "w") as json_file:
        json.dump(models_ranking, json_file, indent=4)
    print(f"Ranking results saved to {json_file_path}")
    return expert_models


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
    return clusters, clustering_result.n_clusters, clustering_result.cluster_centers_


def evaluate_expert_models(expert_models, test, clusters_center):
    predictions = []
    test = scaler.transform(test.iloc[:, 1].values.reshape(-1, 1))
    test_windows, test_labels = util.make_windows(test, WINDOW_SIZE, HORIZON)

    for t in range(len(test_windows)):
        prediction = predict(expert_models, clusters_center, test_windows[t])
        predictions.append(prediction)

        # Convert list to numpy array; assumed shape is (n, 1, 1)
    predictions = np.array(predictions)

    # Squeeze out the last dimension so that predictions become (n, 1)
    predictions = np.squeeze(predictions, axis=2)

    # Alternatively, if you prefer, you can use reshape:
    # predictions = predictions.reshape(predictions.shape[0], predictions.shape[1])

    labels = np.array(test_labels)  # assumed shape is (n, 1)

    mse = np.mean((predictions - labels) ** 2)
    rmse = np.sqrt(mse)

    result = {"rmse :" : rmse}

    return result


def predict(expert_models, clusters_center, test_window):
    min_euc_dist = float('inf')
    current_window = test_window.reshape(1, -1)
    closest_cluster_idx = None

    for idx, center in enumerate(clusters_center):
        center = center.reshape(1, -1)
        euclidean_distance = euclidean_distances(current_window.reshape(1, -1), center)[0][0]
        if euclidean_distance < min_euc_dist:
            min_euc_dist = euclidean_distance
            closest_cluster_idx = idx

    print("---------------------------------------------")
    print(f"Closest cluster index: {closest_cluster_idx}")
    expert_model = expert_models[closest_cluster_idx]
    prediction = expert_model.predict(current_window)
    return prediction


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
            print("Expert models:", expert_models)
            result = evaluate_expert_models(expert_models, test, clusters_center)

            # Save the summed errors (mae, mse, rmse) along with the dataset name.
            save_result_csv(dataset_name, result, csv_filename="expert/results/results.csv")


test_files_path = "test_data"  # Adjust the path as needed
prepare_data(test_files_path)
