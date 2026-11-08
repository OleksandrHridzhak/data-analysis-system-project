"""
Keras MLP forecasting model with input & target standardization, MAE loss, and learning curve tracking.
"""

import os
import joblib
from typing import Dict, Any, Optional, List
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from .base import BaseModel


class KerasMLPModel(BaseModel):
    """
    MLP regressor built in Keras with:
    - Input standardization (StandardScaler)
    - Target logSales standardization (StandardScaler)
    - Dense layers + ReLU + Dropout
    - Linear output layer
    - MAE loss + Adam optimizer
    - Tracks per-epoch train and validation loss
    - Saved in native .keras format alongside scalers
    """

    def __init__(self, model_name: str, params: Optional[Dict[str, Any]] = None):
        if params is None:
            params = {}
        defaults = {
            "hidden_layer_sizes": [512, 256],
            "dropout": 0.1,
            "learning_rate": 1e-4,
            "epochs": 20,
            "batch_size": 512,
            "random_state": 42
        }
        defaults.update(params)
        super().__init__(model_name, defaults)
        self.model_type = "KerasMLP"
        self.input_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.model: Optional[tf.keras.Model] = None
        self.history_: Dict[str, List[float]] = {}

    def _build_model(self, input_dim: int) -> tf.keras.Model:
        seed = int(self.params.get("random_state", 42))
        tf.random.set_seed(seed)
        np.random.seed(seed)

        layers = [tf.keras.layers.Input(shape=(input_dim,))]
        hidden_sizes = self.params.get("hidden_layer_sizes", [512, 256])
        dropout_rate = float(self.params.get("dropout", 0.1))

        for size in hidden_sizes:
            layers.append(tf.keras.layers.Dense(int(size), activation="relu"))
            if dropout_rate > 0.0:
                layers.append(tf.keras.layers.Dropout(dropout_rate))

        layers.append(tf.keras.layers.Dense(1, activation="linear"))

        model = tf.keras.Sequential(layers)
        optimizer = tf.keras.optimizers.Adam(learning_rate=float(self.params.get("learning_rate", 1e-4)))
        model.compile(optimizer=optimizer, loss="mae", metrics=["mae"])
        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> "KerasMLPModel":
        # 1. Scale inputs
        X_train_scaled = self.input_scaler.fit_transform(X_train)
        # 2. Scale target
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.input_scaler.transform(X_val)
            y_val_scaled = self.target_scaler.transform(y_val.reshape(-1, 1)).flatten()
            validation_data = (X_val_scaled, y_val_scaled)

        self.model = self._build_model(input_dim=X_train.shape[1])

        epochs = int(self.params.get("epochs", 20))
        batch_size = int(self.params.get("batch_size", 512))

        history = self.model.fit(
            X_train_scaled,
            y_train_scaled,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )

        train_loss = history.history.get("loss", [])
        val_loss = history.history.get("val_loss", [])
        self.history_ = {
            "train_loss": [float(x) for x in train_loss],
            "val_loss": [float(x) for x in val_loss] if val_loss else []
        }
        self.is_trained = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.model is None:
            raise RuntimeError(f"Model {self.model_name} is not trained.")
        X_scaled = self.input_scaler.transform(X)
        pred_scaled = self.model.predict(X_scaled, verbose=0)
        # Inverse transform to original logSales scale
        pred_log = self.target_scaler.inverse_transform(pred_scaled).flatten()
        return pred_log

    def get_learning_curve(self) -> Optional[Dict[str, List[float]]]:
        if not self.history_:
            return None
        return self.history_

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        # 1. Save Keras architecture & weights in .keras format
        model_path = os.path.join(directory, f"{self.model_name}_mlp.keras")
        self.model.save(model_path)

        # 2. Save scalers and history
        scalers_path = os.path.join(directory, f"{self.model_name}_mlp_scalers.joblib")
        joblib.dump({
            "input_scaler": self.input_scaler,
            "target_scaler": self.target_scaler,
            "history": self.history_,
            "params": self.params
        }, scalers_path)

    @classmethod
    def load(cls, directory: str, model_name: str) -> "KerasMLPModel":
        model_path = os.path.join(directory, f"{model_name}_mlp.keras")
        scalers_path = os.path.join(directory, f"{model_name}_mlp_scalers.joblib")
        if not os.path.exists(model_path) or not os.path.exists(scalers_path):
            raise FileNotFoundError(f"Model artifacts not found for {model_name}")

        scalers_data = joblib.load(scalers_path)
        inst = cls(model_name, scalers_data.get("params", {}))
        inst.input_scaler = scalers_data["input_scaler"]
        inst.target_scaler = scalers_data["target_scaler"]
        inst.history_ = scalers_data.get("history", {})
        inst.model = tf.keras.models.load_model(model_path)
        inst.is_trained = True
        return inst

