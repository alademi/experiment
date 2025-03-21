import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

from Model import util
from Model.first_group_config import ModelBuilder
from Model.util import  perform_clustering

HORIZON = 1
WINDOW_SIZE = 7
MODELS = util.get_models_list()
scaler = StandardScaler()


def train_models(clusters, clusters_no, file_name):

    for i in range(clusters_no):
        train_windows = clusters[i]["train_windows"]
        train_labels = clusters[i]["train_labels"]
        val_windows = clusters[i]["val_windows"]
        val_labels = clusters[i]["val_labels"]

        for model_name in MODELS:
            model_path = f'expert/expert-models/{file_name}/cluster{i + 1}/{model_name}'
            callbacks = [util.create_model_checkpoint(model_path)]

            model_builder = ModelBuilder(model_type=model_name, n_timesteps=WINDOW_SIZE, n_features=HORIZON)
            model = model_builder.build_model()
            model.compile(optimizer='adam', loss='mae')

            model.fit(
                x=train_windows,
                y=train_labels,
                batch_size=128,
                epochs=100,
                verbose=1,
                validation_data=(val_windows, val_labels),
                callbacks=callbacks
            )
           # predictions = model.predict(val_windows)
            #print(predictions)


def cluster_data(train):
   # values = train.iloc[:, 1].to_numpy()
   # time_stamp = train.iloc[:, 0].to_numpy()

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


test_files_path = "../test"
prepare_data(test_files_path)
