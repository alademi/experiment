import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import euclidean_distances
from sklearn.preprocessing import StandardScaler
import joblib

from Model import util
from Model.models_config import ModelBuilder
from Model.util import perform_clustering

HORIZON = 1
WINDOW_SIZE = 7
MODELS = ModelBuilder.get_available_models()
scaler = StandardScaler()


def load_models(clusters_no, dataset):
    """
    Load all models saved in every cluster for the given dataset.
    Returns a list of lists, where each inner list contains tuples (model_name, model)
    for that cluster.
    """
    cluster_models = []
    models_path = f"/Model/first_experiment/expert/expert-models/{dataset}"

    for i in range(clusters_no):
        models = []
        cluster_path = os.path.join(models_path, f"cluster{i + 1}")
        for model_name in MODELS:
            model_dir = os.path.join(cluster_path, model_name)
            # For non-torch models:
            if model_name in ["decision_tree", "random_forest", "xgboost"]:
                checkpoint_file = os.path.join(model_dir, "best_model.pkl")
                if os.path.exists(checkpoint_file):
                    model = joblib.load(checkpoint_file)
                    models.append((model_name, model))
                else:
                    print(f"Checkpoint not found: {checkpoint_file}")
            else:
                # Torch-based models:
                checkpoint_file = os.path.join(model_dir, "best_model.pth")
                if os.path.exists(checkpoint_file):
                    model_builder = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, horizon=HORIZON)
                    model = model_builder.build_model()
                    state_dict = torch.load(checkpoint_file, map_location=torch.device("cpu"), weights_only=True)
                    model.load_state_dict(state_dict)
                    model.eval()
                    models.append((model_name, model))
                else:
                    print(f"Checkpoint not found: {checkpoint_file}")
        cluster_models.append(models)
    return cluster_models


def evaluate_models_by_cluster(test, clusters_no, clusters_center, dataset_name):
    """
    For each test window, assign it to the closest cluster based on euclidean distance
    to the cluster centers. Then, for each model saved in that cluster, generate a prediction.
    The predictions (and the actual next values) are inverse‑transformed to the original scale.
    CSV files are produced for the overall test set and for each cluster.

    The CSV files have rows corresponding to test windows (time points) and columns:
      - "Time": index of the test window (starting at 1),
      - "Cluster": the assigned cluster number,
      - "Actual": the true next value (inverse‑transformed),
      - One column per model (named after the model) with its prediction.

    Returns a dictionary (predictions_by_cluster) that maps each cluster index to its list of rows.
    """
    from sklearn.metrics import euclidean_distances  # In case not imported above.

    # Load expert models per cluster.
    cluster_models = load_models(clusters_no, dataset_name)

    # Create test windows and corresponding labels.
    test_scaled = scaler.transform(test.iloc[:, 1].values.reshape(-1, 1))
    test_windows, test_labels = util.make_windows(test_scaled, WINDOW_SIZE, HORIZON)

    predictions_overall = []  # List of rows for the overall test set.
    predictions_by_cluster = {i: [] for i in range(clusters_no)}  # Rows per cluster.

    for t, window in enumerate(test_windows):
        # Determine the closest cluster.
        current_window = window.reshape(1, -1)
        min_euc_dist = float('inf')
        closest_cluster_idx = None
        for idx, center in enumerate(clusters_center):
            center_reshaped = center.reshape(1, -1)
            dist = euclidean_distances(current_window, center_reshaped)[0][0]
            if dist < min_euc_dist:
                min_euc_dist = dist
                closest_cluster_idx = idx

        # Build a dictionary for this test window.
        row = {}
        row["Time"] = t + 1  # Start time at 1.
        actual_scaled = test_labels[t]
        actual_orig = scaler.inverse_transform(np.array(actual_scaled).reshape(1, 1))[0, 0]
        row["Actual"] = actual_orig

        # For every model in the assigned cluster, get a prediction.
        for (model_name, model) in cluster_models[closest_cluster_idx]:
            # For classical models.
            if model_name in ["decision_tree", "random_forest", "xgboost"]:
                w = window if window.ndim == 2 else window.reshape(1, -1)
                pred = model.predict(w)
                pred = np.array(pred).squeeze()
            else:
                # For torch-based models.
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if model_name == "mlp":
                    if window.ndim == 2 and window.shape[1] == 1:
                        w = np.squeeze(window, axis=1)
                    else:
                        w = window
                    input_tensor = torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(0)
                else:
                    w = window if window.ndim > 1 else np.expand_dims(window, axis=-1)
                    input_tensor = torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(0)
                model.eval()
                with torch.no_grad():
                    output = model(input_tensor)
                    if isinstance(output, tuple):
                        output = output[0]
                    pred = output.cpu().numpy().squeeze()
            # If prediction returns more than one value, select one (e.g., median for "mq-cnn" or first element otherwise).
            if isinstance(pred, np.ndarray) and pred.size > 1:
                if model_name == "mq-cnn":
                    pred = pred[1]
                else:
                    pred = pred[0]
            pred_orig = scaler.inverse_transform(np.array(pred).reshape(1, 1))[0, 0]
            row[model_name] = pred_orig

        row["Cluster"] = closest_cluster_idx + 1  # Use 1-indexed clusters.
        predictions_overall.append(row)
        predictions_by_cluster[closest_cluster_idx].append(row)

    def reorder_df(df):
        print("Before reordering, columns in df:", df.columns.tolist())
        cols = df.columns.tolist()
        new_order = ["Time", "Cluster", "Actual"] + [col for col in cols if col not in ["Time", "Cluster", "Actual"]]
        return df[new_order]

    # Save overall predictions.
    overall_df = pd.DataFrame(predictions_overall)
    overall_df = reorder_df(overall_df)
    overall_file = f"/Model/first_experiment/results/predictions_clusters/{dataset_name}/all_predictions.csv"
    os.makedirs(os.path.dirname(overall_file), exist_ok=True)
    overall_df.to_csv(overall_file, index=False)
    print(f"Overall predictions saved to {overall_file}")

    for cluster_idx, rows in predictions_by_cluster.items():
        print(f"Cluster {cluster_idx} has {len(rows)} rows")  # Debugging statement

        if len(rows) == 0:
            print(f"Warning: No predictions for cluster {cluster_idx}")
            continue  # Skip empty clusters

        cluster_df = pd.DataFrame(rows)
        print(f"Cluster {cluster_idx} DataFrame before reordering:")
        print(cluster_df.head())

        cluster_df = reorder_df(cluster_df)

        cluster_file = f"/Model/first_experiment/results/predictions_clusters/{dataset_name}/cluster{cluster_idx + 1}.csv"
        os.makedirs(os.path.dirname(cluster_file), exist_ok=True)
        cluster_df.to_csv(cluster_file, index=False)
        print(f"Cluster {cluster_idx + 1} predictions saved to {cluster_file}")

    # Return the predictions by cluster (for further processing).
    return predictions_by_cluster


def save_rmse_by_cluster(predictions_by_cluster, dataset_name):
    """
    For each cluster, compute the RMSE (original scale) per model using the predicted and actual values.
    Then, read the global RMSE values from the provided global results file and add a "Global" row.
    Finally, save a CSV file where rows are each cluster (plus a global row) and columns are the models.
    """

    rows_list = []
    for cluster_idx, rows in predictions_by_cluster.items():
        df = pd.DataFrame(rows)
        # Identify model columns (exclude "Time", "Cluster", "Actual").
        model_columns = [col for col in df.columns if col not in ["Time", "Cluster", "Actual"]]
        rmse_dict = {}
        for model in model_columns:
            rmse_val = np.sqrt(np.mean((df[model] - df["Actual"]) ** 2))
            rmse_dict[model] = rmse_val
        row = {"Cluster": f"Cluster {cluster_idx + 1}"}
        row.update(rmse_dict)
        rows_list.append(row)
    cluster_rmse_df = pd.DataFrame(rows_list)

    # Read the global RMSE values from the file.
    global_file = "/Model/first_experiment/results/evaluation/results_old.csv"
    global_df = pd.read_csv(global_file)
    global_row = global_df[global_df["Dataset"] == dataset_name].iloc[0].to_dict()
    del global_row["Dataset"]
    global_rmse = {"Cluster": "Global"}
    global_rmse.update(global_row)

    # Use pd.concat instead of .append
    cluster_rmse_df = pd.concat([cluster_rmse_df, pd.DataFrame([global_rmse])], ignore_index=True)

    rmse_file = f"/Model/first_experiment/results/evaluation/clusters/{dataset_name}_rmse.csv"
    os.makedirs(os.path.dirname(rmse_file), exist_ok=True)
    cluster_rmse_df.to_csv(rmse_file, index=False)
    print(f"RMSE per cluster and global saved to {rmse_file}")


#############################################
# 4. Cluster Data and Prepare Test Windows
#############################################
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
                print(f"Dataset {dataset_name} has more than 20000 rows. Selected first 20000 rows.")
            print(f"Data shape for {dataset_name} after processing: {data.shape}")

            split = int(0.8 * len(data))
            train, test = data[:split], data[split:]

            # Fit scaler on training data.
            scaler.fit(train.iloc[:, 1].values.reshape(-1, 1))

            # Cluster the training data.
            clusters, clusters_no, clusters_center = cluster_data(train)
            # Generate predictions and save overall & per-cluster CSV files.
            predictions_by_cluster = evaluate_models_by_cluster(test, clusters_no, clusters_center, dataset_name)
            # Compute and save RMSE per cluster and global.
            save_rmse_by_cluster(predictions_by_cluster, dataset_name)


if __name__ == "__main__":
    test_files_path = "test"  # Adjust the path as needed.
    prepare_data(test_files_path)
