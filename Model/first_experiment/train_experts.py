import os
import random

import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from Model import util
from Model.models_config import ModelBuilder
from Model.util import perform_clustering


HORIZON = 1
WINDOW_SIZE = 7
BATCH_SIZE = 128
EPOCHS = 50
LR = 0.001
SEED = 42

MODELS = ModelBuilder.get_available_models()
scaler = StandardScaler()


os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def reset_pytorch_configuration():
    """
    Reset PyTorch configuration by clearing CUDA cache and re-setting random seeds.
    This helps avoid interference from previous runs when processing multiple datasets.
    """
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reinitialize random seeds for reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_models(clusters, clusters_no, file_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(clusters_no):
        train_windows = clusters[i]["train_windows"]
        train_labels = clusters[i]["train_labels"]
        val_windows = clusters[i]["val_windows"]
        val_labels = clusters[i]["val_labels"]
        for model_name in MODELS:
            model_path = f'expert/expert-models/{file_name}/cluster{i + 1}/{model_name}'
            os.makedirs(model_path, exist_ok=True)
            # Adjust input shape based on model type.
            # For models like mlp, decision_tree, random_forest, and xgboost, use 2D arrays.
            if model_name in ["mlp", "decision_tree", "random_forest", "xgboost"]:
                if train_windows.ndim == 3 and train_windows.shape[2] == 1:
                    train_windows_input = np.squeeze(train_windows, axis=2)
                    test_windows_input = np.squeeze(val_windows, axis=2)
                else:
                    train_windows_input = train_windows
                    test_windows_input = val_windows
            else:
                # Other models expect 3D input: (batch, WINDOW_SIZE, 1)
                if train_windows.ndim == 2:
                    train_windows_input = np.expand_dims(train_windows, axis=-1)
                    test_windows_input = np.expand_dims(val_windows, axis=-1)
                else:
                    train_windows_input = train_windows
                    test_windows_input = val_windows

            # Non-torch models branch.
            if model_name in ["decision_tree", "random_forest", "xgboost"]:
                model_builder = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, horizon=HORIZON)
                model = model_builder.build_model()
                model.fit(train_windows_input, train_labels)
                checkpoint_path = os.path.join(model_path, "best_model.pkl")
                joblib.dump(model, checkpoint_path)
                print(f"Saved {model_name} model to: {checkpoint_path}")
                predictions = model.predict(test_windows_input)
                predictions = predictions.reshape(-1, 1)
                if predictions.ndim != 2 or predictions.shape[1] != 1:
                    raise ValueError(f"Error in the prediction shape: {predictions.shape}")
                else:
                    print(f"Prediction shape OK: {predictions.shape}")
                # Skip torch-based branch.
                continue

            # Torch-based models branch.
            model_builder = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, horizon=HORIZON)
            model = model_builder.build_model().to(device)
            optimizer = optim.Adam(model.parameters(), lr=LR)
            criterion = nn.L1Loss()

            train_tensor_x = torch.tensor(train_windows_input, dtype=torch.float32)
            train_tensor_y = torch.tensor(train_labels, dtype=torch.float32)
            test_tensor_x = torch.tensor(test_windows_input, dtype=torch.float32)
            test_tensor_y = torch.tensor(val_labels, dtype=torch.float32)

            train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
            test_dataset = TensorDataset(test_tensor_x, test_tensor_y)
            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            best_val_loss = float('inf')
            for epoch in range(1, EPOCHS + 1):
                model.train()
                train_loss = 0.0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * batch_x.size(0)
                train_loss /= len(train_dataset)

                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in test_loader:
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        outputs = model(batch_x)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_x.size(0)
                val_loss /= len(test_dataset)
                print(
                    f"Model: {model_name} | Epoch [{epoch}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_path = os.path.join(model_path, "best_model.pth")
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"  Saved best model to: {checkpoint_path}")


def cluster_data(train_norm):
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
    return clusters, clustering_result.n_clusters

def run_experiment(data_path):
    file_names = os.listdir(data_path)
    for name in file_names:
        if name.lower().endswith('.csv'):
            dataset_name = os.path.splitext(name)[0]
            print("Processing dataset:", dataset_name)

            model_dir = f'expert/expert-models/{dataset_name}'

            # Check if the model directory already exists; if it does, skip this file and log it
            if os.path.isdir(model_dir):
                print(f"{dataset_name} skipped")
                continue

            data = pd.read_csv(os.path.join(data_path, name))
            values = data.iloc[:, 1].values
            if len(values) > 20000:
                data = data.iloc[:20000]
                print(f"Dataset {dataset_name} has more than 20000 rows. Selected the first 20000 rows for processing.")
            print(f"Data shape for {dataset_name} after processing: {data.shape}")
            split = int(0.8 * len(data))
            train, test = data[:split], data[split:]
            values = train.iloc[:, 1].values
            scaler.fit(values.reshape(-1, 1))
            train_norm = scaler.transform(values.reshape(-1, 1))
            clusters, clusters_no = cluster_data(train_norm)
            train_models(clusters, clusters_no, dataset_name)
            reset_pytorch_configuration()

if __name__ == '__main__':
    test_files_path = "test"
    run_experiment(test_files_path)
