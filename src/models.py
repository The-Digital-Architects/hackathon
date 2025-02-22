import joblib
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, StackingRegressor, VotingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge, HuberRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
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
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        y_pred = np.maximum(y_pred, epsilon)
        y_true = np.maximum(y_true, epsilon)
        
        log_array = np.abs(np.log(y_pred/y_true))
        constrained_log_array = np.minimum(log_array, 10)  # Changed min to minimum
        sum_logs = np.sum(constrained_log_array)
        
        return -1 * (1/N * sum_logs)  # Negative because we want to maximize
    
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
    MODELS = {
        'rf': RandomForestRegressor,
        'gb': GradientBoostingRegressor,
        'ada': AdaBoostRegressor,
        'elastic': ElasticNet,
        'xgb': XGBRegressor,
        'lgbm': LGBMRegressor,
        'catboost': CatBoostRegressor,
        'ext': ExtraTreesRegressor,
        'svr': SVR,
        'mlp': MLPRegressor,
        'huber': HuberRegressor,
        'ridge': Ridge,
        'lasso': Lasso
    }

    def _create_model(self):
        params = self.config.get('model_params', {})
        model_type = self.config.get('model_type', 'rf')
        model_class = self.MODELS.get(model_type, RandomForestRegressor)
        return model_class(**params)