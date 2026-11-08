"""
Random Forest forecasting model.
"""

import os
import joblib
from typing import Dict, Any, Optional
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from .base import BaseModel


class RandomForestModel(BaseModel):
    """Wrapper for RandomForestRegressor with standardized interface."""

    def __init__(self, model_name: str, params: Optional[Dict[str, Any]] = None):
        if params is None:
            params = {}
        defaults = {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_leaf": 1,
            "random_state": 42,
            "n_jobs": -1
        }
        defaults.update(params)
        super().__init__(model_name, defaults)
        self.model_type = "RandomForest"
        self.estimator = RandomForestRegressor(
            n_estimators=int(self.params["n_estimators"]),
            max_depth=int(self.params["max_depth"]) if self.params.get("max_depth") is not None else None,
            min_samples_leaf=int(self.params["min_samples_leaf"]),
            random_state=int(self.params.get("random_state", 42)),
            n_jobs=int(self.params.get("n_jobs", -1))
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> "RandomForestModel":
        self.estimator.fit(X_train, y_train)
        self.is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError(f"Model {self.model_name} is not trained.")
        return self.estimator.predict(X)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        if not self.is_trained or not hasattr(self.estimator, "feature_importances_"):
            return None
        return self.estimator.feature_importances_

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.model_name}_rf.joblib")
        joblib.dump(self.estimator, path)

    @classmethod
    def load(cls, directory: str, model_name: str) -> "RandomForestModel":
        path = os.path.join(directory, f"{model_name}_rf.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        estimator = joblib.load(path)
        inst = cls(model_name)
        inst.estimator = estimator
        inst.is_trained = True
        return inst

