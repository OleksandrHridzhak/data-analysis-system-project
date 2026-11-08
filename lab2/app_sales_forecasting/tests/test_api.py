"""
Integration tests for FastAPI endpoints and error statuses (404, 409, 422).
"""

import pytest
from fastapi.testclient import TestClient
from api.main import app
from api import main as main_module
from api.service import ForecastingService

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_service(tmp_path):
    # Initialize service with temporary models directory
    main_module.service = ForecastingService(models_dir=str(tmp_path))


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "available_models" in data


def test_data_info_endpoint():
    response = client.get("/data/info")
    assert response.status_code == 200
    data = response.json()
    assert data["total_stores"] == 100
    assert "all_candidate_features" in data
    assert "Customers" not in data["all_candidate_features"]


def test_predict_without_trained_model_returns_409():
    """Predicting before any model is trained must return 409 Conflict."""
    payload = {
        "model_name": "non_existent",
        "record": {"Store": 1, "Date": "2015-05-01"}
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 409


def test_invalid_model_type_returns_422():
    """Unsupported model type should return 422 Unprocessable Entity."""
    payload = {
        "model_type": "InvalidModelType",
        "model_name": "test_inv"
    }
    response = client.post("/train", json=payload)
    assert response.status_code == 422


def test_full_api_workflow_train_predict_forecast_delete():
    # 1. Train small RF model
    train_payload = {
        "model_type": "RandomForest",
        "model_name": "api_test_rf",
        "features": ["Promo", "Store", "weekday"],
        "hyperparameters": {"n_estimators": 5, "max_depth": 3, "random_state": 42}
    }
    train_resp = client.post("/train", json=train_payload)
    assert train_resp.status_code == 200
    train_data = train_resp.json()
    assert train_data["model_name"] == "api_test_rf"
    assert "metrics" in train_data
    assert "train" in train_data["metrics"]
    assert "validation" in train_data["metrics"]
    assert "test" in train_data["metrics"]

    # 2. Check /models
    models_resp = client.get("/models")
    assert models_resp.status_code == 200
    assert len(models_resp.json()) == 1

    # 3. Check /models/api_test_rf
    detail_resp = client.get("/models/api_test_rf")
    assert detail_resp.status_code == 200
    assert detail_resp.json()["model_name"] == "api_test_rf"

    # 4. Unknown model returns 404
    missing_resp = client.get("/models/unknown_model_123")
    assert missing_resp.status_code == 404

    # 5. Predict single (open day)
    pred_payload = {
        "model_name": "api_test_rf",
        "record": {
            "Store": 1,
            "Date": "2015-05-01",
            "Open": 1,
            "Promo": 1
        }
    }
    pred_resp = client.post("/predict", json=pred_payload)
    assert pred_resp.status_code == 200
    preds = pred_resp.json()["predictions"]
    assert len(preds) == 1
    assert preds[0]["predicted_sales"] > 0

    # 6. Predict single (closed day -> must be 0)
    pred_closed_payload = {
        "model_name": "api_test_rf",
        "record": {
            "Store": 1,
            "Date": "2015-05-03",
            "Open": 0,
            "Promo": 0
        }
    }
    pred_closed_resp = client.post("/predict", json=pred_closed_payload)
    assert pred_closed_resp.status_code == 200
    assert pred_closed_resp.json()["predictions"][0]["predicted_sales"] == 0.0

    # 7. Predict batch
    batch_payload = {
        "model_name": "api_test_rf",
        "records": [
            {"Store": 1, "Date": "2015-05-01", "Open": 1},
            {"Store": 2, "Date": "2015-05-01", "Open": 1}
        ]
    }
    batch_resp = client.post("/predict/batch", json=batch_payload)
    assert batch_resp.status_code == 200
    assert len(batch_resp.json()["predictions"]) == 2

    # 8. Forecast endpoint
    fc_payload = {
        "model_name": "api_test_rf",
        "stores": [1],
        "start_date": "2015-05-01",
        "end_date": "2015-05-07"
    }
    fc_resp = client.post("/forecast", json=fc_payload)
    assert fc_resp.status_code == 200
    fc_data = fc_resp.json()
    assert len(fc_data["forecasts"]) == 7
    assert fc_data["has_actuals"] is True

    # 9. Delete model
    del_resp = client.delete("/models/api_test_rf")
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True

    # Verify deleted
    assert client.get("/models/api_test_rf").status_code == 404
