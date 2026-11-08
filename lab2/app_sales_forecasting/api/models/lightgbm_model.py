"""
LightGBM forecasting model with early stopping and learning curve tracking.
"""

import os
import joblib
from typing import Dict, Any, Optional, List
import numpy as np
from lightgbm import LGBMRegressor, early_stopping, record_evaluation
from .base import BaseModel


class LightGBMModel(BaseModel):
    """Wrapper for LGBMRegressor with early stopping and loss history tracking."""

    def __init__(self, model_name: str, params: Optional[Dict[str, Any]] = None):
        if params is None:
            params = {}
        defaults = {
            "n_estimators": 1000,
            "learning_rate": 0.1,
            "num_leaves": 100,
            "max_depth": 5,
            "colsample_bytree": 0.7,
            "random_state": 42,
            "n_jobs": -1
        }
        defaults.update(params)
        super().__init__(model_name, defaults)
        self.model_type = "LightGBM"
        self.evals_result_: Dict[str, Dict[str, List[float]]] = {}
        
        self.estimator = LGBMRegressor(
            n_estimators=int(self.params["n_estimators"]),
            learning_rate=float(self.params["learning_rate"]),
            num_leaves=int(self.params["num_leaves"]),
            max_depth=int(self.params["max_depth"]) if self.params.get("max_depth") != -1 else None,
            colsample_bytree=float(self.params["colsample_bytree"]),
            random_state=int(self.params.get("random_state", 42)),
            n_jobs=int(self.params.get("n_jobs", -1)),
            verbose=-1
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> "LightGBMModel":
        callbacks = []
        eval_set = [(X_train, y_train)]
        eval_names = ["train"]

        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            eval_names.append("valid")
            callbacks.append(early_stopping(stopping_rounds=50, verbose=False))

        eval_dict: Dict[str, Dict[str, List[float]]] = {}
        callbacks.append(record_evaluation(eval_dict))

        self.estimator.fit(
            X_train, y_train,
            eval_set=eval_set,
            eval_names=eval_names,
            eval_metric="rmse",
            callbacks=callbacks
        )
        self.evals_result_ = eval_dict
        self.is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError(f"Model {self.model_name} is not trained.")
        return self.estimator.predict(X)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        if not self.is_trained or not hasattr(self.estimator, "feature_importances_"):
            return None
        return self.estimator.feature_importances_.astype(float)

    def get_learning_curve(self) -> Optional[Dict[str, List[float]]]:
        if not self.evals_result_:
            return None
        train_loss = self.evals_result_.get("train", {}).get("rmse", [])
        val_loss = self.evals_result_.get("valid", {}).get("rmse", [])
        return {
            "train_loss": [float(x) for x in train_loss],
            "val_loss": [float(x) for x in val_loss]
        }

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.model_name}_lgb.joblib")
        joblib.dump({"estimator": self.estimator, "evals_result": self.evals_result_}, path)

    @classmethod
    def load(cls, directory: str, model_name: str) -> "LightGBMModel":
        path = os.path.join(directory, f"{model_name}_lgb.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        data = joblib.load(path)
        inst = cls(model_name)
        inst.estimator = data["estimator"]
        inst.evals_result_ = data.get("evals_result", {})
        inst.is_trained = True
        return inst
