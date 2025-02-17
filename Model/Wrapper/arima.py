from statsmodels.tsa.arima.model import ARIMA


class ARIMAModelWrapper:
    """
    A simple wrapper for a statsmodels ARIMA model that mimics a Keras‑like interface.
    Note that ARIMA models are not neural networks, so they do not integrate with
    Keras training loops.
    """

    def __init__(self, order=(1, 0, 0)):
        self.order = order
        self.model = None
        self.fitted_model = None

    def fit(self, y, **kwargs):
        """
        Fits the ARIMA model.

        Parameters:
            y (array-like): The 1D time series data.
            kwargs: Additional keyword arguments passed to the ARIMA fit method.
        """
        self.model = ARIMA(y, order=self.order)
        self.fitted_model = self.model.fit(**kwargs)
        return self

    def predict(self, start=None, end=None, **kwargs):
        """
        Makes predictions using the fitted ARIMA model.

        Parameters:
            start (int): The starting index of the forecast.
            end (int): The ending index of the forecast.
            kwargs: Additional keyword arguments passed to the predict method.

        Returns:
            np.ndarray: The forecasted values.
        """
        if self.fitted_model is None:
            raise ValueError("The ARIMA model must be fitted before prediction.")
        return self.fitted_model.predict(start=start, end=end, **kwargs)

    def summary(self):
        """
        Returns a summary of the fitted ARIMA model.
        """
        if self.fitted_model is None:
            raise ValueError("The ARIMA model must be fitted before calling summary.")
        return self.fitted_model.summary()