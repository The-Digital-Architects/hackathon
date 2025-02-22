import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier


class BaseModel:
    def __init__(self, config = None):
        self.config = config or {}
        self.model: BaseEstimator = self._create_model()
    
    def _create_model(self):
        raise NotImplementedError
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        joblib.dump(self.model, path)
    
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
