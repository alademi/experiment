# classical_forecasting_models.py

import numpy as np
import inspect
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from statsmodels.tsa.arima.model import ARIMA


class ARIMAWrapper:
    """
    A wrapper for statsmodels' ARIMA that adapts to a sliding-window supervised setting.

    Instead of training on the full series, this wrapper fits a separate ARIMA model for each
    sliding window (subsequence) and forecasts the next value.

    Parameters:
      order (tuple): The (p,d,q) order of the ARIMA model. Default is (1, 0, 0).
    """

    def __init__(self, order=(1, 0, 0)):
        self.order = order

    def fit(self, X, y):
        """
        In this implementation, fit does nothing because ARIMA is re‐fitted for every window.
        This method is included to mimic a scikit‑learn regressor interface.
        """
        return self

    def predict(self, X):
        """
        Given a 2D array X where each row is a sliding window (subsequence) of a time series,
        fit an ARIMA model on each window and forecast one step ahead.

        Parameters:
          X: numpy array of shape (n_samples, window_size)

        Returns:
          forecasts: numpy array of shape (n_samples,) containing the next value prediction for each window.
        """
        forecasts = []
        for row in X:
            # Fit an ARIMA model on the current window (row)
            # row is a 1D array of length window_size.
            model = ARIMA(row, order=self.order)
            model_fit = model.fit()
            fc = model_fit.forecast(steps=1)
            forecasts.append(fc[0])
        return np.array(forecasts)


class ModelBuilder:
    """
    A builder class to create classical forecasting/regression models that operate on sliding-window data.

    Supported model types:
      - "arima": Uses ARIMAWrapper to fit each window and forecast the next value.
      - "decision-tree" or "dtr": Uses DecisionTreeRegressor.
      - "random-forest" or "rafo": Uses RandomForestRegressor.
      - "gradient-boosted-trees" or "gbt": Uses GradientBoostingRegressor.

    For the tree-based models, X should be a 2D array where each row is a window (subsequence)
    of the time series and y is a 1D array with the next value (label) for each window.

    Additional parameters can be passed via kwargs.
    """

    def __init__(self, model_type="arima", **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None

    def _filter_kwargs(self, constructor):
        """
        Filters self.kwargs to only include parameters accepted by the constructor.
        """
        valid_params = inspect.signature(constructor.__init__).parameters
        return {k: v for k, v in self.kwargs.items() if k in valid_params}

    def build_model(self):
        if self.model_type == "arima":
            # For ARIMAWrapper, we only need the order.
            order = self.kwargs.get("order", (1, 0, 0))
            self.model = ARIMAWrapper(order=order)
        elif self.model_type in ("decision-tree", "dtr"):
            filtered_kwargs = self._filter_kwargs(DecisionTreeRegressor)
            self.model = DecisionTreeRegressor(**filtered_kwargs)
        elif self.model_type in ("random-forest", "rafo"):
            filtered_kwargs = self._filter_kwargs(RandomForestRegressor)
            self.model = RandomForestRegressor(**filtered_kwargs)
        elif self.model_type in ("gradient-boosted-trees", "gbt"):
            filtered_kwargs = self._filter_kwargs(GradientBoostingRegressor)
            self.model = GradientBoostingRegressor(**filtered_kwargs)
        else:
            raise ValueError(
                "Invalid model type. Choose from 'arima', 'decision-tree' (or 'dtr'), "
                "'random-forest' (or 'rafo'), or 'gradient-boosted-trees' (or 'gbt')."
            )
        return self.model

    @staticmethod
    def get_available_models():
        """
        Returns a list of available model names.
        """
        return ["decision-tree", "random-forest", "gradient-boosted-trees"]


# -------------------------
# Example Usage
# -------------------------
if __name__ == '__main__':
    # Create a dummy univariate time series of length 110.
    # We use a sliding window approach where each window (length=10) predicts the next value.
    np.random.seed(42)
    time_series = np.random.randn(110)
    window_size = 10

    # Create supervised learning data: each row in X is a window, y is the next value.
    X = np.array([time_series[i: i + window_size] for i in range(len(time_series) - window_size)])
    y = time_series[window_size:]
    print("X shape:", X.shape, "y shape:", y.shape)  # Expected: (100, 10) and (100,)

    # ARIMA example:
    builder_arima = ModelBuilder(model_type="arima", order=(1, 1, 1))
    arima_model = builder_arima.build_model()
    # Fit method does nothing in this implementation.
    arima_model.fit(None, None)
    # Predict next value for each window in X using ARIMA
    arima_forecasts = arima_model.predict(X)
    print("\nARIMA forecasts (first 5):", arima_forecasts[:5])

    # Decision Tree example:
    # Extra parameter n_timesteps (or any unsupported parameter) will be filtered out.
    builder_dt = ModelBuilder(model_type="decision-tree", n_timesteps=window_size, max_depth=5)
    dt_model = builder_dt.build_model()
    dt_model.fit(X, y)
    dt_preds = dt_model.predict(X)
    print("\nDecision Tree Predictions (first 5):", dt_preds[:5])

    # Random Forest example:
    builder_rf = ModelBuilder(model_type="random-forest", n_estimators=100, n_timesteps=window_size)
    rf_model = builder_rf.build_model()
    rf_model.fit(X, y)
    rf_preds = rf_model.predict(X)
    print("\nRandom Forest Predictions (first 5):", rf_preds[:5])

    # Gradient Boosted Trees example:
    builder_gbt = ModelBuilder(model_type="gradient-boosted-trees", n_estimators=100, n_timesteps=window_size)
    gbt_model = builder_gbt.build_model()
    gbt_model.fit(X, y)
    gbt_preds = gbt_model.predict(X)
    print("\nGradient Boosted Trees Predictions (first 5):", gbt_preds[:5])
