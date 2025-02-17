import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import your PyTorch ModelBuilder and utility functions.
from Model.models_config import ModelBuilder
import Model.util as util  # assuming util.make_windows and util.get_models_list exist

# Hyperparameters
HORIZON = 1
WINDOW_SIZE = 7
BATCH_SIZE = 128
EPOCHS = 100
LR = 0.001

MODELS = ModelBuilder.get_available_models()
scaler = StandardScaler()

def train_models(train_windows, train_labels, test_windows, test_labels, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loop over each model type.
    for model_name in MODELS:
        print(f"\nTraining model: {model_name}")

        # Define model path for saving checkpoints.
        model_path = os.path.join('base', 'base-models', dataset, model_name)
        os.makedirs(model_path, exist_ok=True)

        # --- Preprocess input shapes ---
        if model_name == "mlp":
            # MLP expects 2D input.
            if train_windows.ndim == 3 and train_windows.shape[2] == 1:
                train_windows_input = np.squeeze(train_windows, axis=2)
                test_windows_input = np.squeeze(test_windows, axis=2)
            else:
                train_windows_input = train_windows
                test_windows_input = test_windows
        else:
            # Other models expect 3D input (batch, WINDOW_SIZE, HORIZON).
            if train_windows.ndim == 2:
                train_windows_input = np.expand_dims(train_windows, axis=-1)
                test_windows_input = np.expand_dims(test_windows, axis=-1)
            else:
                train_windows_input = train_windows
                test_windows_input = test_windows

        # --- Build the model ---
        model_builder = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, n_features=HORIZON)
        model = model_builder.build_model().to(device)

        # --- Define optimizer and loss function ---
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.L1Loss()  # MAE

        # --- Prepare DataLoaders ---
        train_tensor_x = torch.tensor(train_windows_input, dtype=torch.float32)
        train_tensor_y = torch.tensor(train_labels, dtype=torch.float32)
        test_tensor_x = torch.tensor(test_windows_input, dtype=torch.float32)
        test_tensor_y = torch.tensor(test_labels, dtype=torch.float32)

        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        test_dataset = TensorDataset(test_tensor_x, test_tensor_y)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        best_val_loss = float('inf')

        # --- Training loop ---
        for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                # If outputs is a tuple (e.g., for DeepAR), unpack it and use the mean for the loss.
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * batch_x.size(0)

            train_loss /= len(train_dataset)

            # --- Validation ---
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

            print(f"Model: {model_name} | Epoch [{epoch}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

            # --- Checkpoint: Save best model ---
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(model_path, "best_model.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  Saved best model to: {checkpoint_path}")

        # --- Evaluate predictions shape on test set ---
        model.eval()
        with torch.no_grad():
            test_inputs = torch.tensor(test_windows_input, dtype=torch.float32).to(device)
            predictions = model(test_inputs)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
        if predictions.ndim != 2 or predictions.shape[1] != 1:
            raise ValueError(f"Error in the prediction shape: {predictions.shape}")
        else:
            print(f"Prediction shape OK: {predictions.shape}")

def prepare_data(data_path):
    file_names = os.listdir(data_path)
    for name in file_names:
        if name.lower().endswith('.csv'):
            dataset_name = os.path.splitext(name)[0]
            print(f"\nProcessing dataset: {dataset_name}")

            data = pd.read_csv(os.path.join(data_path, name))
            split = int(0.8 * len(data))
            train_df, test_df = data[:split], data[split:]

            # Assuming the series is in the second column (index 1)
            train_series = train_df.iloc[:, 1].values.reshape(-1, 1)
            test_series = test_df.iloc[:, 1].values.reshape(-1, 1)

            scaler.fit(train_series)
            train_scaled = scaler.transform(train_series)
            test_scaled = scaler.transform(test_series)

            # Create sliding windows.
            train_windows, train_labels = util.make_windows(train_scaled, WINDOW_SIZE, horizon=HORIZON)
            test_windows, test_labels = util.make_windows(test_scaled, WINDOW_SIZE, horizon=HORIZON)

            # Train models on this dataset.
            train_models(train_windows, train_labels, test_windows, test_labels, dataset_name)

if __name__ == '__main__':
    test_files_path = "test-models"  # update to your CSV folder
    prepare_data(test_files_path)
