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

from Model.models_config import ModelBuilder
import Model.util as util


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



def train_models(train_windows, train_labels, test_windows, test_labels, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model_name in MODELS:
        print(f"\nTraining model: {model_name}")
        model_path = os.path.join('base', 'base-models', dataset, model_name)

        if os.path.exists(model_path):
            print(f"Skipping: {dataset}/{model_name}")
            continue

        os.makedirs(model_path, exist_ok=True)

        # Adjust input shape based on model type.
        # For models like mlp, decision_tree, random_forest, and xgboost, use 2D arrays.
        if model_name in ["mlp", "decision_tree", "random_forest", "xgboost"]:
            if train_windows.ndim == 3 and train_windows.shape[2] == 1:
                train_windows_input = np.squeeze(train_windows, axis=2)
                test_windows_input = np.squeeze(test_windows, axis=2)
            else:
                train_windows_input = train_windows
                test_windows_input = test_windows
        else:
            # Other models expect 3D input: (batch, WINDOW_SIZE, 1)
            if train_windows.ndim == 2:
                train_windows_input = np.expand_dims(train_windows, axis=-1)
                test_windows_input = np.expand_dims(test_windows, axis=-1)
            else:
                train_windows_input = train_windows
                test_windows_input = test_windows

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
            continue

        # Torch-based models branch.
        model_builder = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, horizon=HORIZON)
        model = model_builder.build_model().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.L1Loss()

        train_tensor_x = torch.tensor(train_windows_input, dtype=torch.float32)
        train_tensor_y = torch.tensor(train_labels, dtype=torch.float32)
        test_tensor_x = torch.tensor(test_windows_input, dtype=torch.float32)
        test_tensor_y = torch.tensor(test_labels, dtype=torch.float32)

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
            print(f"Model: {model_name} | Epoch [{epoch}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(model_path, "best_model.pth")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  Saved best model to: {checkpoint_path}")

        model.eval()
        with torch.no_grad():
            test_inputs = torch.tensor(test_windows_input, dtype=torch.float32).to(device)
            predictions = model(test_inputs)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            # Check if the model output has a quantile dimension and select the median quantile.
            if predictions.ndim == 3 and predictions.shape[2] > 1:
                predictions = predictions[:, :, 1]
        if predictions.ndim != 2 or predictions.shape[1] != 1:
            raise ValueError(f"Error in the prediction shape: {predictions.shape}")
        else:
            print(f"Prediction shape OK: {predictions.shape}")

    print("Training complete.")

def prepare_data(data_path):
    file_names = os.listdir(data_path)
    for name in file_names:
        if name.lower().endswith('.csv'):
            dataset_name = os.path.splitext(name)[0]



            print(f"\nProcessing dataset: {dataset_name}")
            data = pd.read_csv(os.path.join(data_path, name))
            values = data.iloc[:, 1].values
            if len(values) > 20000:
                data = data.iloc[:20000]
                print(f"Dataset {dataset_name} has more than 20000 rows. Selected the first 20000 rows for processing.")
            print(f"Data shape for {dataset_name} after processing: {data.shape}")

            split = int(0.8 * len(data))
            train_df, test_df = data[:split], data[split:]
            train_series = train_df.iloc[:, 1].values.reshape(-1, 1)
            test_series = test_df.iloc[:, 1].values.reshape(-1, 1)

            scaler.fit(train_series)
            train_scaled = scaler.transform(train_series)
            test_scaled = scaler.transform(test_series)

            train_windows, train_labels = util.make_windows(train_scaled, WINDOW_SIZE, horizon=HORIZON)
            test_windows, test_labels = util.make_windows(test_scaled, WINDOW_SIZE, horizon=HORIZON)

            train_models(train_windows, train_labels, test_windows, test_labels, dataset_name)

            # Reset PyTorch settings after processing each dataset
            reset_pytorch_configuration()


if __name__ == '__main__':
    test_files_path = "test"
    prepare_data(test_files_path)
