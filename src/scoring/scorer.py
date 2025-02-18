"""Abstract scorer class from which other scorers inherit from."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Scorer(ABC):
    """Abstract base class for all scorers."""

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: Any) -> float:
        """Calculate the score."""
        pass

    @abstractmethod
    def __str__(self) -> str:
        """Return the name of the scorer."""
        pass
