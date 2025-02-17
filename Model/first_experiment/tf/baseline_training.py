import os

import pandas as pd
from sklearn.preprocessing import StandardScaler

import Model.util as util
from Model.first_group_config import ModelBuilder

HORIZON = 1
WINDOW_SIZE = 7
MODELS = util.get_models_list()
scaler = StandardScaler()


def train_models(train_windows, train_labels, test_windows, test_labels, dataset):
    # Train each model in MODELS
    for model_name in MODELS:
        # Keep model_path as is
        model_path = f'base/base-models/{dataset}/{model_name}'
        # Use the updated callback
        callbacks = [util.create_model_checkpoint(model_path)]

        # Build and compile the model
        model_builder = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, n_features=HORIZON)
        model = model_builder.build_model()

        model.compile(optimizer='adam', loss='mae')

        # Train the model with checkpointing (saving best weights)
        model.fit(
            x=train_windows,
            y=train_labels,
            batch_size=128,
            epochs=100,
            verbose=1,
            validation_data=(test_windows, test_labels),
            callbacks=callbacks
        )
        predictions = model.predict(test_windows)
        if (predictions.shape[1] != 1):
            raise(f"Error in the prediction shape : {predictions.shape}")

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
            train = scaler.transform(train.iloc[:, 1].values.reshape(-1, 1))
            train_windows, train_labels = util.make_windows(train, WINDOW_SIZE, horizon=HORIZON)
            test = scaler.transform(test.iloc[:, 1].values.reshape(-1, 1))
            test_windows, test_labels = util.make_windows(test, WINDOW_SIZE)
            train_models(train_windows, train_labels, test_windows, test_labels, dataset_name)


test_files_path = "../test"
prepare_data(test_files_path)