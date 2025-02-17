import csv
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

# Import your utility functions and the PyTorch ModelBuilder.
from Model import util
from Model.models_config import ModelBuilder

HORIZON = 1
WINDOW_SIZE = 7
MODELS = ModelBuilder.get_available_models()
scaler = StandardScaler()

RESULTS_FILE = "/Users/aalademi/PycharmProjects/experiment/Model/first_experiment/base/results/base_results.csv"


def load_models(dataset):
    """
    Rebuild each model using ModelBuilder and load its saved checkpoint.
    The checkpoint is assumed to be saved as 'best_model.pth' in:
      /.../base/classical-models/{dataset}/{model_name}/best_model.pth
    """
    models_list = []
    for model_name in MODELS:
        checkpoint_path = f"/Users/aalademi/PycharmProjects/experiment/Model/first_experiment/base/base-models/{dataset}/{model_name}/best_model.pth"
        # Rebuild the model. For MLP, ModelBuilder ignores n_features.
        model_builder = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, n_features=HORIZON)
        model = model_builder.build_model()
        # Load the state dictionary using weights_only=True to avoid the pickle warning.
        state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        models_list.append((model_name, model))
    return models_list


def evaluate(model, model_name, subsequences, labels):
    """
    Evaluates the model on each provided window (subsequence) one at a time.

    For the MLP model:
      - If the input has an extra dimension (shape (WINDOW_SIZE, 1)), it is squeezed to (WINDOW_SIZE,)
      - Then a batch dimension is added to yield shape (1, WINDOW_SIZE)

    For all other models (which expect 3D input):
      - If a subsequence is 1D (shape: (WINDOW_SIZE,)), we add a new axis to get (WINDOW_SIZE, 1)
      - Then a batch dimension is added to yield (1, WINDOW_SIZE, 1)

    The function loops over each window individually, computes the model's prediction, and finally
    calculates the RMSE over all predictions.
    """
    device = next(model.parameters()).device
    predictions = []

    for i in range(len(subsequences)):
        subseq = subsequences[i]

        if model_name == "mlp":
            # MLP expects a 2D input: (batch, WINDOW_SIZE)
            # If subseq is shape (WINDOW_SIZE, 1), squeeze it.
            if subseq.ndim == 2 and subseq.shape[1] == 1:
                subseq = np.squeeze(subseq, axis=1)
            # If subseq is already 1D, do nothing.
            input_tensor = torch.tensor(subseq, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            # Other models expect 3D input: (batch, WINDOW_SIZE, HORIZON)
            # If subseq is 1D (i.e., shape: (WINDOW_SIZE,)), add a new axis to become (WINDOW_SIZE, 1)
            if subseq.ndim == 1:
                subseq = np.expand_dims(subseq, axis=-1)
            # If subseq is 2D but not of shape (WINDOW_SIZE, HORIZON), you might add checks here.
            input_tensor = torch.tensor(subseq, dtype=torch.float32, device=device).unsqueeze(0)

        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
            # Unpack tuple if necessary (e.g., for DeepAR, output is (mean, sigma)).
            if isinstance(output, tuple):
                output = output[0]
        predictions.append(output.cpu().numpy().squeeze())

    predictions = np.array(predictions)
    print("##########################")
    print(f"Model :{model_name}")
    print(predictions)
    print("##########################")

    labels = np.array(labels).squeeze()
    mse = np.mean((predictions - labels) ** 2)
    return np.sqrt(mse)


def save_to_csv(dataset_name, results):
    file_exists = os.path.isfile(RESULTS_FILE)
    fieldnames = ["Dataset"] + MODELS

    with open(RESULTS_FILE, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

    print(f"Results saved to {RESULTS_FILE}")


def evaluate_models(test_norm, dataset_name):
    test_windows, test_labels = util.make_windows(test_norm, WINDOW_SIZE, HORIZON)
    models_list = load_models(dataset_name)
    results = {"Dataset": dataset_name}

    for model_name, model in models_list:
        rmse = evaluate(model, model_name, test_windows, test_labels)
        results[model_name] = rmse  # Store RMSE under the model's name

    save_to_csv(dataset_name, results)


def prepare_data(data_path):
    file_names = os.listdir(data_path)
    for name in file_names:
        if name.lower().endswith('.csv'):
            dataset_name = os.path.splitext(name)[0]
            print(f"Processing dataset: {dataset_name}")

            data = pd.read_csv(os.path.join(data_path, name))
            split = int(0.8 * len(data))
            train, test = data[:split], data[split:]

            scaler.fit(train.iloc[:, 1].values.reshape(-1, 1))
            test_norm = scaler.transform(test.iloc[:, 1].values.reshape(-1, 1))

            evaluate_models(test_norm, dataset_name)


if __name__ == '__main__':
    test_files_path = "test-models"
    prepare_data(test_files_path)
