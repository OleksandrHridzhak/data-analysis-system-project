"""
API Client for interacting with the Sales Forecasting FastAPI microservice.
Strictly communicates via HTTP REST requests. Never imports ML code.
"""

import os
from typing import Dict, Any, List, Optional
import requests

DEFAULT_API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


class SalesForecastingAPIClient:
    """HTTP client for the sales forecasting microservice."""

    def __init__(self, base_url: str = DEFAULT_API_URL, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """Checks API health and returns available models."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                return resp.json()
            return {"status": "error", "code": resp.status_code, "available_models": []}
        except Exception as e:
            return {"status": "unreachable", "error": str(e), "available_models": []}

    def get_data_info(self) -> Dict[str, Any]:
        """Fetches dataset metadata and tunable parameters."""
        resp = requests.get(f"{self.base_url}/data/info", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def train_model(
        self,
        model_type: str,
        model_name: Optional[str] = None,
        features: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Sends POST request to train a model."""
        payload = {
            "model_type": model_type,
            "model_name": model_name,
            "features": features,
            "hyperparameters": hyperparameters or {}
        }
        resp = requests.post(f"{self.base_url}/train", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def list_models(self) -> List[Dict[str, Any]]:
        """Retrieves list of trained models."""
        resp = requests.get(f"{self.base_url}/models", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_model(self, model_name: str) -> Dict[str, Any]:
        """Retrieves specific model details."""
        resp = requests.get(f"{self.base_url}/models/{model_name}", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def delete_model(self, model_name: str) -> Dict[str, Any]:
        """Deletes a model."""
        resp = requests.delete(f"{self.base_url}/models/{model_name}", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def predict_single(self, model_name: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """Predicts for single record."""
        payload = {"model_name": model_name, "record": record}
        resp = requests.post(f"{self.base_url}/predict", json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def predict_batch(self, model_name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predicts for batch of records."""
        payload = {"model_name": model_name, "records": records}
        resp = requests.post(f"{self.base_url}/predict/batch", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def forecast(
        self,
        model_name: str,
        stores: List[int],
        start_date: str,
        end_date: str,
        promo_switch: Optional[int] = None,
        scenario_overrides: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Requests daily forecast with optional scenario simulation."""
        payload = {
            "model_name": model_name,
            "stores": stores,
            "start_date": start_date,
            "end_date": end_date,
            "promo_switch": promo_switch,
            "scenario_overrides": scenario_overrides or []
        }
        resp = requests.post(f"{self.base_url}/forecast", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
