import csv
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from Model import util
from Model.models_config import ModelBuilder
import joblib

HORIZON = 1
WINDOW_SIZE = 7
MODELS = ModelBuilder.get_available_models()
scaler = StandardScaler()

RESULTS_FILE = "/Model/first_experiment/results/evaluation/results_old.csv"


def load_models(dataset):
    """
    Rebuild each model using ModelBuilder and load its saved checkpoint.
    For decision_tree, random_forest, and xgboost the checkpoint is assumed to be saved as 'best_model.pkl'
    in: /.../base/base-models/{dataset}/{model_name}/best_model.pkl
    For torch-based models, the checkpoint is 'best_model.pth'.
    """
    models_list = []
    for model_name in MODELS:
        if model_name in ["decision_tree", "random_forest", "xgboost"]:
            checkpoint_path = f"/Model/first_experiment/base/base-models/{dataset}/{model_name}/best_model.pkl"
            model = joblib.load(checkpoint_path)
            models_list.append((model_name, model))
        else:
            checkpoint_path = f"/Model/first_experiment/base/base-models/{dataset}/{model_name}/best_model.pth"
            model_builder = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, horizon=HORIZON)
            model = model_builder.build_model()
            # Load the state dictionary with weights_only=True to avoid pickle warnings.
            state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            models_list.append((model_name, model))
    return models_list


def evaluate(model, model_name, subsequences, labels):
    """
    Evaluate the model on each subsequence individually, compute predictions,
    and return both the normalized RMSE and the inverse-transformed predictions.
    """
    predictions = []

    for i in range(len(subsequences)):
        subseq = np.array(subsequences[i])
        if hasattr(model, 'predict'):
            # Classical model branch.
            if subseq.ndim == 1:
                subseq = subseq.reshape(1, -1)
            elif subseq.ndim == 2 and subseq.shape[1] == 1:
                subseq = np.squeeze(subseq, axis=-1).reshape(1, -1)
            pred = model.predict(subseq)
            pred = np.squeeze(pred)
        else:
            # Torch-based model branch.
            device = next(model.parameters()).device
            if model_name == "mlp":
                # MLP expects 2D input: (batch, WINDOW_SIZE)
                if subseq.ndim == 2 and subseq.shape[1] == 1:
                    subseq = np.squeeze(subseq, axis=1)
                input_tensor = torch.tensor(subseq, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                # Other torch models expect 3D input: (batch, WINDOW_SIZE, 1)
                if subseq.ndim == 1:
                    subseq = np.expand_dims(subseq, axis=-1)
                input_tensor = torch.tensor(subseq, dtype=torch.float32, device=device).unsqueeze(0)
            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                if isinstance(output, tuple):
                    output = output[0]
            pred = output.cpu().numpy().squeeze()

        # If the model is "mq-cnn", select the median quantile (assumed at index 1).
        if model_name == "mq-cnn":
            pred = np.atleast_1d(pred)
            if pred.shape[0] == 3:
                pred = pred[1]

        predictions.append(pred)
        print(f"Subsequence {i}: Prediction = {pred}, True label = {np.array(labels)[i]}")

    predictions = np.array(predictions)
    labels = np.array(labels).squeeze()

    # Inverse-transform to the original scale.
    predictions_orig = scaler.inverse_transform(predictions.reshape(-1, 1)).squeeze()
    labels_orig = scaler.inverse_transform(labels.reshape(-1, 1)).squeeze()

    mse_orig = np.mean((predictions_orig - labels_orig) ** 2)
    rmse_orig = np.sqrt(mse_orig)

    print("##########################")
    print(f"Model: {model_name}")
    print(f"Predictions (original scale): {predictions_orig}")
    print("##########################")

    # Return both the rmse and the inverse-transformed predictions.
    return rmse_orig, predictions_orig


def save_to_csv(results):
    file_exists = os.path.isfile(RESULTS_FILE)
    fieldnames = ["Dataset"] + MODELS

    with open(RESULTS_FILE, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

    print(f"Results saved to {RESULTS_FILE}")


def evaluate_models_and_save_predictions(test_norm, dataset_name):
    """
    This function creates windows from the normalized test data, evaluates all models,
    collects the actual values and predictions, and saves them to a CSV file.
    """
    test_windows, test_labels = util.make_windows(test_norm, WINDOW_SIZE, HORIZON)
    # Inverse-transform test labels for saving.
    test_labels_orig = scaler.inverse_transform(np.array(test_labels).reshape(-1, 1)).squeeze()
    models_list = load_models(dataset_name)

    # Dictionary to hold predictions for each model.
    predictions_dict = {"Cluster" : -1, "Actual": test_labels_orig}
    results = {"Dataset": dataset_name}

    for model_name, model in models_list:
        rmse, predictions_orig = evaluate(model, model_name, test_windows, test_labels)
        results[model_name] = rmse  # Store normalized RMSE for logging.
        predictions_dict[model_name] = predictions_orig

    # Save the evaluation metrics using your existing function.
    save_to_csv(results)

    # Create a DataFrame with actual values and model predictions.
    predictions_df = pd.DataFrame(predictions_dict)
    predictions_file = f"/Model/first_experiment/results/predictions_upd/{dataset_name}_predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to {predictions_file}")


def evaluate_models(test_norm, dataset_name):
    evaluate_models_and_save_predictions(test_norm, dataset_name)


def prepare_data(data_path):
    file_names = os.listdir(data_path)
    for name in file_names:
        if name.lower().endswith('.csv'):
            dataset_name = os.path.splitext(name)[0]
            print(f"Processing dataset: {dataset_name}")

            data = pd.read_csv(os.path.join(data_path, name))
            values = data.iloc[:, 1].values

            # If more than 20,000 rows, select the first 20,000.
            if len(values) > 20000:
                data = data.iloc[:20000]
                print(f"Dataset {dataset_name} has more than 20000 rows. Selected first 20000 rows.")

            print(f"Data shape for {dataset_name} after processing: {data.shape}")

            split = int(0.8 * len(data))
            train, test = data[:split], data[split:]

            # Fit scaler on training data.
            scaler.fit(train.iloc[:, 1].values.reshape(-1, 1))

            test_norm = scaler.transform(test.iloc[:, 1].values.reshape(-1, 1))

            evaluate_models(test_norm, dataset_name)


if __name__ == '__main__':
    test_files_path = "test"
    prepare_data(test_files_path)
