from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from .base import BaseModel

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

# Register available models
MODEL_REGISTRY = {
    'random_forest': RandomForest,
    'adaboost': AdaBoost,
    'neural_network': NeuralNetwork,
}

__all__ = ['BaseModel', 'RandomForest', 'AdaBoost', 'NeuralNetwork', 'MODEL_REGISTRY']
