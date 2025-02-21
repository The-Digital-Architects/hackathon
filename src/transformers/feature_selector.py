import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from .base import BaseTransformer

class FeatureSelector(BaseTransformer):
    def __init__(self, config=None):
        super().__init__(config)
        self.params = self.config.get('params', {})
        self.selector = None
        
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'FeatureSelector':
        n_features = min(self.params.get('n_features', 10), X.shape[1])
        self.selector = SelectKBest(score_func=f_classif, k=n_features)
        self.selector.fit(X, y)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.selector.transform(X)
