"""
Fast unit tests for RandomForest, LightGBM, and Keras MLP models.
Tests fit, predict, save, load, feature importances, and learning curves.
"""

import os
import pytest
import numpy as np
from api.models import (
    RandomForestModel,
    LightGBMModel,
    KerasMLPModel,
    calculate_metrics
)


@pytest.fixture
def dummy_data():
    np.random.seed(42)
    X = np.random.randn(50, 10)
    y = 5.0 + 0.5 * X[:, 0] + np.random.normal(0, 0.1, size=50)
    return X, y


def test_calculate_metrics():
    y_true = np.array([5.0, 6.0, 7.0])
    y_pred = np.array([5.1, 5.9, 7.2])
    m = calculate_metrics(y_true, y_pred)
    assert "rmse_logSales" in m
    assert "mae_Sales" in m
    assert "rmse_Sales" in m
    assert "wape_Sales" in m
    assert "r2_Sales" in m
    assert m["rmse_logSales"] > 0


def test_random_forest_roundtrip(dummy_data, tmp_path):
    X, y = dummy_data
    model = RandomForestModel("rf_test", {"n_estimators": 5, "max_depth": 3, "random_state": 42})
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    
    fi = model.get_feature_importance()
    assert fi is not None
    assert len(fi) == X.shape[1]

    # Save & Load
    model.save(str(tmp_path))
    loaded = RandomForestModel.load(str(tmp_path), "rf_test")
    preds_loaded = loaded.predict(X)
    assert np.allclose(preds, preds_loaded)


def test_lightgbm_roundtrip(dummy_data, tmp_path):
    X, y = dummy_data
    model = LightGBMModel("lgb_test", {"n_estimators": 10, "num_leaves": 15, "max_depth": 3, "random_state": 42})
    model.fit(X, y, X_val=X[:10], y_val=y[:10])
    preds = model.predict(X)
    assert len(preds) == len(y)

    lc = model.get_learning_curve()
    assert lc is not None
    assert "train_loss" in lc

    # Save & Load
    model.save(str(tmp_path))
    loaded = LightGBMModel.load(str(tmp_path), "lgb_test")
    preds_loaded = loaded.predict(X)
    assert np.allclose(preds, preds_loaded)


def test_keras_mlp_roundtrip(dummy_data, tmp_path):
    X, y = dummy_data
    model = KerasMLPModel("mlp_test", {"hidden_layer_sizes": [32, 16], "epochs": 2, "batch_size": 16, "random_state": 42})
    model.fit(X, y, X_val=X[:10], y_val=y[:10])
    preds = model.predict(X)
    assert len(preds) == len(y)

    lc = model.get_learning_curve()
    assert lc is not None
    assert "train_loss" in lc
    assert len(lc["train_loss"]) == 2

    # Save & Load
    model.save(str(tmp_path))
    loaded = KerasMLPModel.load(str(tmp_path), "mlp_test")
    preds_loaded = loaded.predict(X)
    assert np.allclose(preds, preds_loaded, atol=1e-3)

