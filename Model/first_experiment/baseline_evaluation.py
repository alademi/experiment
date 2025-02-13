import csv
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models

from Model import util

HORIZON = 1
WINDOW_SIZE = 7
MODELS = util.get_models_list()
scaler = StandardScaler()

RESULTS_FILE = "/Users/aalademi/PycharmProjects/experiment/Model/first_experiment/base/results/results.csv"


def load_models(dataset):
    models_list = []
    for model_name in MODELS:
        model_path = f"/Users/aalademi/PycharmProjects/experiment/Model/first_experiment/base/base-models/{dataset}/{model_name}"
        model = models.load_model(model_path)
        models_list.append((model_name, model))
    return models_list


def evaluate(model, subsequences, labels):
    predictions = []
    for t in range(len(subsequences)):
        current_window = subsequences[t].reshape(1, -1)
        prediction = model.predict(current_window)
        predictions.append(prediction)

    predictions = np.array(predictions).squeeze(axis=2)
    labels = np.array(labels)

    mse = np.mean((predictions - labels) ** 2)
    return np.sqrt(mse)


def save_to_csv(dataset_name, results):
    file_exists = os.path.isfile(RESULTS_FILE)

    # Define column names: ["Dataset", "conv", "conv-lstm", "mlp", "lstm"]
    fieldnames = ["Dataset"] + MODELS

    with open(RESULTS_FILE, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write header only if the file does not exist
        if not file_exists:
            writer.writeheader()

        # Write the current dataset results
        writer.writerow(results)

    print(f"Results saved to {RESULTS_FILE}")


def evaluate_models(test_norm, dataset_name):
    test_windows, test_labels = util.make_windows(test_norm, WINDOW_SIZE, HORIZON)
    models = load_models(dataset_name)

    # Create results dictionary with dataset name
    results = {"Dataset": dataset_name}

    for model_name, model in models:
        rmse = evaluate(model, test_windows, test_labels)
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


test_files_path = "test_data"  # Adjust as needed
prepare_data(test_files_path)
