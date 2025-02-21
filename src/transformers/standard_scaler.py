import numpy as np
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
from .base import BaseTransformer

class StandardScaler(BaseTransformer):
    def __init__(self, config=None):
        super().__init__(config)
        self.scaler = SklearnStandardScaler()
    
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> 'StandardScaler':
        """Fit ignores y parameter."""
        self.scaler.fit(X)
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)
