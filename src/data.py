from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataHandler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scaler = StandardScaler()
        
    def load_data(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess data."""
        df = pd.read_csv(path)
        
        # Separate features and target
        target_col = self.config.get('target_column', 'target')
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Basic preprocessing
        X = self._preprocess(X)
        
        return X, y.values
    
    def _preprocess(self, X: pd.DataFrame) -> np.ndarray:
        """Basic preprocessing."""
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Scale features
        return self.scaler.fit_transform(X)
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets."""
        test_size = self.config.get('test_size', 0.2)
        random_state = self.config.get('random_state', 42)
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
