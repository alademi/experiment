from statsmodels.tsa.arima.model import ARIMA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


class ModelBuilder:
    def __init__(self, model_type="autoregressive", order=(1, 0, 0), n_estimators=100, max_depth=None):
        self.model_type = model_type
        self.model = None
        self.order = order
        self.n_estimators = n_estimators  # For ensemble models
        self.max_depth = max_depth  # For tree-based models

    def build_model(self):
        if self.model_type == "autoregressive":
            self.model = self._build_autoregressive_model()
        elif self.model_type == "arima":
            self.model = self._build_arima_model()
        elif self.model_type == "decision_tree":
            self.model = self._build_decision_tree_model()
        elif self.model_type == "random_forest":
            self.model = self._build_random_forest_model()
        elif self.model_type == "gradient_boosting":
            self.model = self._build_gradient_boosting_model()
        else:
            raise ValueError("Invalid model type. Choose from available model types.")
        return self.model

    def _build_autoregressive_model(self):
        return LinearRegression()

    def _build_arima_model(self):
        return lambda y_train: ARIMA(y_train, order=self.order).fit()

    def _build_decision_tree_model(self):
        return DecisionTreeRegressor(max_depth=self.max_depth)

    def _build_random_forest_model(self):
        return RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth)

    def _build_gradient_boosting_model(self):
        return GradientBoostingRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth)
