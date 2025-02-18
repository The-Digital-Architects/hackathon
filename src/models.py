from typing import Any, Dict
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier

class BaseModel:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.model: BaseEstimator = self._create_model()
    
    def _create_model(self) -> BaseEstimator:
        raise NotImplementedError
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        joblib.dump(self.model, path)
    
    @classmethod
    def load(cls, path: str) -> 'BaseModel':
        instance = cls()
        instance.model = joblib.load(path)
        return instance

class RandomForest(BaseModel):
    def _create_model(self) -> BaseEstimator:
        params = self.config.get('model_params', {})
        return RandomForestClassifier(**params)

class AdaBoost(BaseModel):
    def _create_model(self) -> BaseEstimator:
        params = self.config.get('model_params', {})
        return AdaBoostClassifier(**params)

class NeuralNetwork(BaseModel):
    def _create_model(self) -> BaseEstimator:
        params = self.config.get('model_params', {})
        return MLPClassifier(**params)
