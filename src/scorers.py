from typing import Any
import numpy as np
from abc import ABC, abstractmethod


class Scorer(ABC):
    """Abstract base class for all scorers."""

    @abstractmethod
    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate the score."""
        pass

    @abstractmethod
    def __str__(self):
        """Return the name of the scorer."""
        pass


class Accuracy(Scorer):
    """MSE scorer."""

    def __init__(self, name):
        """Initialize the scorer with a name."""
        self.name = name

    def __call__(self, y_true, y_pred, **kwargs):
        """Calculate the score."""
        # Apply a threshold of 0.5 to the predictions
        # Sqeeze the predictions to a 1D array
        y_pred = np.squeeze(y_pred)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0

        # Calculate the accuracy
        return np.mean(y_true == y_pred)

    def __str__(self):
        """Return the name of the scorer."""
        return self.name