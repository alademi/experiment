import json
import numpy as np
import tensorflow as tf
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from Model import util
from Model.first_group_config import ModelBuilder
from Model.util import compute_clusters_no

HORIZON = 1
WINDOW_SIZE = 7
MODELS = ["conv", "conv-lstm", "mlp", "lstm"]
scaler = StandardScaler()


def init_model(file_name, train_windows, train_labels, test_windows, test_labels):
    model_path = f'model_experiments/{MODELS}/{file_name}'
    # Initialize ModelBuilder with desired model type and input parameters
    model_builder = ModelBuilder(model_type="model8", n_timesteps=WINDOW_SIZE, n_features=HORIZON)
    # Build and compile the model using ModelBuilder
    model = model_builder.build_model()
    # Compile the model (even if already compiled in ModelBuilder, adjust if needed)
    model.compile(optimizer='adam', loss='mae')

    # Fit the model to the training data
    model.fit(
        train_windows,
        train_labels,
        batch_size=128,
        epochs=100,
        verbose=1,
        validation_data=(test_windows, test_labels),
        callbacks=[util.create_model_checkpoint("base_model", model_path)]
    )

    return model


def eval_val(model, subsequences, labels):
    """
    Evaluate the model on the validation set.
    Here we loop over each window and predict. (Adjust if you prefer batch prediction.)
    """
    predictions = []
    for t in range(len(subsequences)):
        prediction = model.predict(subsequences[t])
        predictions.append(prediction)
    mse = tf.metrics.mean_squared_error(labels, predictions)
    return tf.sqrt(mse)


def perform_clustering(subsequences):
    # If the input is 3D, reshape it to 2D
    if subsequences.ndim > 2:
        n_samples = subsequences.shape[0]
        subsequences = subsequences.reshape(n_samples, -1)

    clusters_no = compute_clusters_no(subsequences)
    kmeans = KMeans(n_clusters=clusters_no, random_state=42)
    kmeans.fit(subsequences)
    return kmeans


def train_models(clusters, clusters_no, file_name):
    """
    For each cluster:
      - Train every model defined in MODELS using the training data.
      - Save each model's best weights using a checkpoint callback.
      - After training, load the saved model and evaluate it on the validation set.
      - Rank all models by RMSE and select the best one.
      - Store the ranking and the best model's details.
    """
    best_models = {"file_name": file_name, "clusters": {}}

    # Loop over each cluster
    for i in range(clusters_no):
        train_windows = clusters[i]["train_windows"]
        train_labels = clusters[i]["train_labels"]
        val_windows = clusters[i]["val_windows"]
        val_labels = clusters[i]["val_labels"]

        # Dictionary to store each model's validation RMSE in the current cluster
        model_results = {}

        # Train each model in MODELS
        for model_name in MODELS:
            model_path = f'model_experiments/cluster{i + 1}/{model_name}/{file_name}'

            # Build and compile the model
            model_builder = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, n_features=HORIZON)
            model = model_builder.build_model()
            model.compile(optimizer='adam', loss='mae')

            # Train the model with checkpointing (saving best weights)
            model.fit(
                train_windows,
                train_labels,
                batch_size=128,
                epochs=100,
                verbose=1,
                validation_data=(val_windows, val_labels),
                callbacks=[util.create_model_checkpoint(model_name, model_path)]
            )

            # Load the best saved model weights into a fresh model instance
            best_model = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, n_features=HORIZON).build_model()
            best_model.load_weights(model_path)
            best_model.compile(optimizer='adam', loss='mae')

            # Evaluate the loaded model on the validation set
            rmse_tensor = eval_val(best_model, val_windows, val_labels)
            rmse_val = rmse_tensor.numpy() if hasattr(rmse_tensor, "numpy") else rmse_tensor

            model_results[model_name] = rmse_val

        # Identify the best model (with the lowest RMSE)
        best_model_name = min(model_results, key=model_results.get)
        best_model_path = f'model_experiments/cluster{i + 1}/{best_model_name}/{file_name}'

        # (Optional) Reload the best model to confirm its performance
        best_model_final = ModelBuilder(model_type=best_model_name, n_timesteps=WINDOW_SIZE, n_features=HORIZON).build_model()
        best_model_final.load_weights(best_model_path)
        best_model_final.compile(optimizer='adam', loss='mae')
        final_rmse_tensor = eval_val(best_model_final, val_windows, val_labels)
        final_rmse = final_rmse_tensor.numpy() if hasattr(final_rmse_tensor, "numpy") else final_rmse_tensor

        # Create a ranking (sorted list of models by RMSE)
        sorted_ranking = sorted(model_results.items(), key=lambda x: x[1])
        sorted_models = [item[0] for item in sorted_ranking]
        sorted_rmse = [item[1] for item in sorted_ranking]

        # Store the ranking and best model details for this cluster
        best_models["clusters"][f"cluster_{i + 1}"] = {
            "Models Ranking": sorted_models,
            "RMSE Ranking": sorted_rmse,
            "Best Model": best_model_name,
            "Best Model RMSE": final_rmse,
            "All Models RMSE": model_results
        }

    # Save the overall results to a JSON file
    json_file_path = f"expert_models/{file_name}.json"
    with open(json_file_path, "w") as json_file:
        json.dump(best_models, json_file, indent=4)

    print(f"Best model results saved to {json_file_path}")
    return best_models


def cluster_train(train):
    values = train.iloc[:, 1].to_numpy()
    time_stamp = train.iloc[:, 0].to_numpy()

    train_norm = scaler.transform(train.iloc[:, 1].values.reshape(-1, 1))
    train_windows, train_labels = util.make_windows(train_norm, WINDOW_SIZE, HORIZON)
    clustering_result = perform_clustering(train_windows)

    # Build clusters based on KMeans labels
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

    train_models(clusters, clustering_result.n_clusters, "Bitcoin-Price")


def evaluate_models(test):
    # You can implement evaluation on test data similar to the validation process.
    pass


def prepare_data(data_path):
    data = pd.read_csv(data_path)
    split = int(0.8 * len(data))
    train, test = data[:split], data[split:]
    scaler.fit(train.iloc[:, 1].values.reshape(-1, 1))
    cluster_train(train)
    evaluate_models(test)


# Call the processing function with the path to your test CSV file
test_files_path = "test_files/Bitcoin_price.csv"  # Adjust the path as needed
prepare_data(test_files_path)
