import os
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
import Model.models_config as config

HORIZON = 1
WINDOW_SIZE = 7
MODELS = ModelBuilder.get_available_models()  # This now returns a list of available model names.
scaler = StandardScaler()


def train_models(clusters, clusters_no, file_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for i in range(clusters_no):
        # Get cluster training and validation data
        train_windows = clusters[i]["train_windows"]
        train_labels = clusters[i]["train_labels"]
        val_windows = clusters[i]["val_windows"]
        val_labels = clusters[i]["val_labels"]

        for model_name in MODELS:
            model_path = f'expert/expert-models/{file_name}/cluster{i + 1}/{model_name}'
            os.makedirs(model_path, exist_ok=True)

            # Adjust the input shape:
            # For non-MLP models, we need the data to be 3D: (samples, WINDOW_SIZE, HORIZON)
            if model_name != "mlp":
                if train_windows.ndim == 2:
                    train_windows_model = np.expand_dims(train_windows, axis=-1)
                    val_windows_model = np.expand_dims(val_windows, axis=-1)
                else:
                    train_windows_model = train_windows
                    val_windows_model = val_windows
            else:
                # MLP expects 2D input.
                train_windows_model = train_windows
                val_windows_model = val_windows

            # Convert NumPy arrays to PyTorch tensors.
            train_x = torch.tensor(train_windows_model, dtype=torch.float32)
            train_y = torch.tensor(train_labels, dtype=torch.float32)
            val_x = torch.tensor(val_windows_model, dtype=torch.float32)
            val_y = torch.tensor(val_labels, dtype=torch.float32)

            # Create TensorDatasets and DataLoaders.
            train_dataset = TensorDataset(train_x, train_y)
            val_dataset = TensorDataset(val_x, val_y)
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

            # Build the model using your PyTorch ModelBuilder.
            model_builder = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, n_features=HORIZON)
            model = model_builder.build_model().to(device)

            # Set up the optimizer and loss function (MAE).
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.L1Loss()

            best_val_loss = float('inf')
            epochs = 100

            for epoch in range(1, epochs + 1):
                model.train()
                train_loss = 0.0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    # Unpack output if model returns a tuple (e.g., DeepAR returns (mean, sigma)).
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * batch_x.size(0)
                train_loss /= len(train_dataset)

                # Validation phase.
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        outputs = model(batch_x)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_x.size(0)
                val_loss /= len(val_dataset)

                print(f'Cluster {i + 1}, Model: {model_name}, Epoch [{epoch}/{epochs}], '
                      f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

                # Save checkpoint if validation loss improves.
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint_file = os.path.join(model_path, "best_model.pth")
                    torch.save(model.state_dict(), checkpoint_file)
                    print(f"  Saved best model to {checkpoint_file}")


def cluster_data(train):
    # Get values from the second column.
    values = train.iloc[:, 1].values
    train_norm = scaler.transform(values.reshape(-1, 1))
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


def prepare_data(data_path):
    file_names = os.listdir(data_path)
    for name in file_names:
        if name.lower().endswith('.csv'):
            dataset_name = os.path.splitext(name)[0]
            print("Processing dataset:", dataset_name)
            data = pd.read_csv(os.path.join(data_path, name))
            split = int(0.8 * len(data))
            train, test = data[:split], data[split:]
            values = train.iloc[:, 1].values
            scaler.fit(values.reshape(-1, 1))
            clusters, clusters_no = cluster_data(train)
            train_models(clusters, clusters_no, dataset_name)


if __name__ == '__main__':
    test_files_path = "test-models"
    prepare_data(test_files_path)
