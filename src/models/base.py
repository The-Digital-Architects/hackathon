from typing import Dict, Any
import numpy as np
from sklearn.base import BaseEstimator

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
