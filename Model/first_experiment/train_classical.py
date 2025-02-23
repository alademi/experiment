import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import math
import torch

# Import our classical model builder.
# This file (classical_model_config.py) should contain the ModelBuilder class
# that builds models for: 'arima', 'decision-tree', 'random-forest', and 'gradient-boosted-trees'.
from Model.classical_model_config import ModelBuilder
import Model.util as util  # This module should provide util.make_windows

# Hyperparameters for sliding windows
HORIZON = 1
WINDOW_SIZE = 7

# List of model types to train
MODELS = ["decision-tree", "random-forest", "gradient-boosted-trees"]

# Instantiate a scaler
scaler = StandardScaler()


def train_classical_models(train_windows, train_labels, dataset):
    """
    Trains each classical forecasting model on sliding-window training data,
    then saves the trained model to disk in .pth format.

    For tree-based models, the training data should be 2D:
      - Each row is a window (subsequence) of length WINDOW_SIZE.
    """
    # Ensure windows are 2D: (n_samples, window_size)
    if train_windows.ndim == 3 and train_windows.shape[2] == 1:
        X_train = np.squeeze(train_windows, axis=2)
    else:
        X_train = train_windows

    for model_name in MODELS:
        print(f"Training model: {model_name}")
        kwargs = {}
        if model_name == "arima":
            # Set ARIMA order; adjust as needed.
            kwargs["order"] = (1, 1, 1)
        builder = ModelBuilder(model_type=model_name, **kwargs)
        model = builder.build_model()

        # For ARIMA, our wrapper is designed to operate on a per-window basis.
        # The fit() method exists for interface consistency and does nothing.
        if model_name == "arima":
            model.fit(None, None)
        else:
            model.fit(X_train, train_labels)

        # Prepare the directory to save the model
        save_dir = os.path.join("base/base-models", dataset, model_name)
        if not os.path.exists(save_dir):
            print(f"Directory {save_dir} does not exist. Creating directory...")
            os.makedirs(save_dir)
        else:
            print(f"Directory {save_dir} already exists.")

        # Save the trained model using torch.save in .pth format.
        model_filename = os.path.join(save_dir, "best_model.pth")
        torch.save(model, model_filename)
        print(f"Saved model {model_name} to {model_filename}")


def evaluate_classical_models(test_windows, test_labels, dataset):
    """
    Loads the saved classical models and evaluates them on the test set.

    Each test window (subsequence) is processed one-by-one:
      For each window, the model's predict() method is called,
      and the overall RMSE is computed.
    """
    # Ensure test windows are 2D: (n_samples, window_size)
    if test_windows.ndim == 3 and test_windows.shape[2] == 1:
        X_test = np.squeeze(test_windows, axis=2)
    else:
        X_test = test_windows

    results = {}
    for model_name in MODELS:
        model_filename = os.path.join("/Users/aalademi/PycharmProjects/experiment/Model/first_experiment/base/base-models", dataset, model_name, "best_model.pth")
        print(f"Loading model {model_name} from {model_filename}")
        model = torch.load(model_filename)

        # Evaluate point-by-point.
        predictions = []
        for i in range(X_test.shape[0]):
            window = X_test[i:i + 1, :]  # shape (1, window_size)
            pred = model.predict(window)
            predictions.append(pred[0])
        predictions = np.array(predictions)
        rmse = math.sqrt(mean_squared_error(test_labels, predictions))
        print(f"Model: {model_name} | RMSE on test set: {rmse:.4f}")
        results[model_name] = rmse
    return results


def prepare_data(data_path):
    """
    For each CSV file in the provided folder:
      - Reads the time series,
      - Splits into training and test sets,
      - Scales the data,
      - Creates sliding windows and labels,
      - Trains the classical models, and
      - Evaluates them point-by-point.
    Also, collects the evaluation results for each dataset and saves them in a CSV file.
    """
    all_results = {}  # Dictionary to store results for each dataset
    file_names = os.listdir(data_path)
    for name in file_names:
        if name.lower().endswith('.csv'):
            dataset = os.path.splitext(name)[0]
            print(f"\nProcessing dataset: {dataset}")
            data = pd.read_csv(os.path.join(data_path, name))
            split_index = int(0.8 * len(data))
            train_df = data[:split_index]
            test_df = data[split_index:]

            # Assume the univariate time series is in the second column (index 1).
            train_series = train_df.iloc[:, 1].values.reshape(-1, 1)
            test_series = test_df.iloc[:, 1].values.reshape(-1, 1)

            scaler.fit(train_series)
            train_scaled = scaler.transform(train_series)
            test_scaled = scaler.transform(test_series)

            # Create sliding windows using util.make_windows.
            train_windows, train_labels = util.make_windows(train_scaled, WINDOW_SIZE, horizon=HORIZON)
            test_windows, test_labels = util.make_windows(test_scaled, WINDOW_SIZE, horizon=HORIZON)

            # Train models and save them.
            train_classical_models(train_windows, train_labels, dataset)
            # Evaluate models by loading them and predicting point-by-point.
            results = evaluate_classical_models(test_windows, test_labels, dataset)
            all_results[dataset] = results

    # Prepare the directory to save results
    results_dir = os.path.join("/Users/aalademi/PycharmProjects/experiment/Model/first_experiment/base", "base-models", "results")
    if not os.path.exists(results_dir):
        print(f"Directory {results_dir} does not exist. Creating directory...")
        os.makedirs(results_dir)
    else:
        print(f"Directory {results_dir} already exists.")

    # Save all_results to a CSV file.
    save_filename = os.path.join(results_dir, "classical_results.csv")


    df = pd.DataFrame.from_dict(all_results, orient='index')
    df.index.name = "Dataset"
    df.reset_index(inplace=True)
    df.to_csv(save_filename, index=False)
    print(f"\nResults saved to {save_filename}")


if __name__ == '__main__':
    data_path = "test_files"  # Update this to your CSV folder path.
    prepare_data(data_path)
