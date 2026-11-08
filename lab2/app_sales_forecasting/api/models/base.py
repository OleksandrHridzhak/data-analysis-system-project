"""
Base interface and metric calculation for all sales forecasting models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def calculate_metrics(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> Dict[str, float]:
    """
    Computes required metrics:
    - RMSE on logSales scale
    - MAE, RMSE, WAPE, R2 on original Sales scale (expm1)
    """
    # Prevent extreme or negative values
    y_pred_log = np.clip(y_pred_log, 0.0, 15.0)
    
    # 1. logSales RMSE
    rmse_log = float(root_mean_squared_error(y_true_log, y_pred_log))
    
    # 2. Sales scale
    y_true_sales = np.expm1(y_true_log)
    y_pred_sales = np.expm1(y_pred_log)
    
    mae_sales = float(mean_absolute_error(y_true_sales, y_pred_sales))
    rmse_sales = float(root_mean_squared_error(y_true_sales, y_pred_sales))
    
    sum_actual = float(np.sum(y_true_sales))
    if sum_actual > 0:
        wape_sales = float(np.sum(np.abs(y_true_sales - y_pred_sales)) / sum_actual)
    else:
        wape_sales = 0.0
        
    r2_sales = float(r2_score(y_true_sales, y_pred_sales))
    
    return {
        "rmse_logSales": round(rmse_log, 4),
        "mae_Sales": round(mae_sales, 2),
        "rmse_Sales": round(rmse_sales, 2),
        "wape_Sales": round(wape_sales, 4),
        "r2_Sales": round(r2_sales, 4)
    }


class BaseModel(ABC):
    """Abstract base model ensuring all three models share the exact same interface."""

    def __init__(self, model_name: str, params: Dict[str, Any]):
        self.model_name = model_name
        self.params = params
        self.model_type = "base"
        self.is_trained = False
        self.features: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.baseline_metrics: Dict[str, Any] = {}
        self.training_time_sec: float = 0.0

    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> "BaseModel":
        """Fits model on preprocessed training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns predictions on logSales scale."""
        pass

    @abstractmethod
    def save(self, directory: str) -> None:
        """Persists model artifact to disk."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, directory: str, model_name: str) -> "BaseModel":
        """Loads model artifact from disk."""
        pass

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Returns raw feature importances across preprocessed columns, or None."""
        return None

    def get_learning_curve(self) -> Optional[Dict[str, List[float]]]:
        """Returns per-iteration train and validation loss, or None."""
        return None
