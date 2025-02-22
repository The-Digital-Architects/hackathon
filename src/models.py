import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier
import numpy as np

class BaseModel:
    def __init__(self, config=None):
        self.config = config or {}
        self.model = self._create_model()
    
    def _create_model(self):
        raise NotImplementedError
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        joblib.dump(self.model, path)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        N = len(y_true)
        log_array = np.abs(np.log(y_pred/y_true))
        constrained_log_array = np.min(log_array, 10)
        sum_logs = np.sum(constrained_log_array)
        return 1/N * sum_logs
    
    @classmethod
    def load(cls, path):
        instance = cls()
        instance.model = joblib.load(path)
        return instance


class RandomForest(BaseModel):
    def _create_model(self):
        params = self.config.get('model_params', {})
        return RandomForestClassifier(**params)


class AdaBoost(BaseModel):
    def _create_model(self):
        params = self.config.get('model_params', {})
        return AdaBoostClassifier(**params)


class NeuralNetwork(BaseModel):
    def _create_model(self):
        params = self.config.get('model_params', {})
        return MLPClassifier(**params)

class WildfirePredictor(BaseModel):
    def _create_model(self, n_estimators=100, random_state=42):
        params = self.config.get('model_params', {})
        return RandomForestRegressor(**params)