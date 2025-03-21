import numpy as np
from keras.layers import Add
from tensorflow.keras import Sequential, Model, regularizers
from tensorflow.keras.layers import Lambda, Dense, Conv1D, AveragePooling1D, LSTM, Dropout, Flatten, Input
import tensorflow as tf



class ModelBuilder:
    def __init__(self, model_type="conv", n_timesteps=None, n_features=None):
        """
        Parameters:
            model_type (str): Type of model to build. Options include 'conv', 'conv-lstm',
                              'mlp', 'lstm', 'ar', and 'decision-tree'.
            n_timesteps (int): Number of time steps in the input.
            n_features (int): Number of features per time step.
        """
        self.model_type = model_type
        self.model = None
        self.n_timesteps = n_timesteps
        self.n_features = n_features

    def build_model(self):
        """
        Builds the selected model architecture.
        For neural network models, a Keras model is returned.
        For Decision Tree, a scikit-learn object is returned.
        """
        if self.model_type == "conv":
            self.model = self._build_model1()
        elif self.model_type == "conv-lstm":
            self.model = self._build_model2()
        elif self.model_type == "mlp":
            self.model = self._build_model3()
        elif self.model_type == "lstm":
            if self.n_timesteps is None or self.n_features is None:
                raise ValueError("n_timesteps and n_features must be provided for the LSTM model.")
            self.model = self._build_model4()
        elif self.model_type == "ar":  # Autoregressive linear model
            self.model = self._build_ar_model()
        else:
            raise ValueError(
                "Invalid model type. Choose from 'conv', 'conv-lstm', 'mlp', 'lstm', 'ar'")

        return self.model  # Make sure to return the model!

    def _build_model1(self):
        model = Sequential(name="conv")
        model.add(Input(shape=(self.n_timesteps, self.n_features)))
        model.add(Conv1D(filters=32, kernel_size=1, strides=1, activation='relu'))
        model.add(Flatten())
        model.add(Dense(10))
        model.add(Dense(20))
        model.add(Dense(30))
        model.add(Dense(units=1))
        return model

    def _build_model2(self):
        model = Sequential(name="conv-lstm")
        model.add(Input(shape=(self.n_timesteps, self.n_features)))
        model.add(Conv1D(filters=32, kernel_size=1, activation='relu'))
        model.add(AveragePooling1D(pool_size=1))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        return model

    def _build_model3(self):
        model = Sequential(name="mlp")
        model.add(Dense(10, input_shape=(self.n_timesteps,)))
        model.add(Dense(20))
        model.add(Dense(30))
        model.add(Dense(1))
        return model

    def _build_model4(self):
        model = Sequential(name="lstm")
        model.add(LSTM(100, activation='relu', input_shape=(self.n_timesteps, self.n_features)))
        model.add(Dense(1))
        return model

    def _build_ar_model(self):
        """
        Builds a simple autoregressive linear model.
        This model expects lagged inputs (i.e., previous time steps) and outputs the next value.
        """
        model = Sequential(name="autoregressive")
        model.add(Input(shape=(self.n_timesteps, self.n_features)))
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))
        return model


