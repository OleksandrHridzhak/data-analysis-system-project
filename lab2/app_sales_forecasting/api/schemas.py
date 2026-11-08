"""
Pydantic schemas for the Sales Forecasting FastAPI microservice.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "sales-forecasting-api"
    available_models: List[str] = Field(default_factory=list)


class DataInfoResponse(BaseModel):
    store_ids: List[int]
    total_stores: int
    train_date_range: Dict[str, str]
    val_date_range: Dict[str, str]
    test_date_range: Dict[str, str]
    candidate_numerical_features: List[str]
    candidate_categorical_features: List[str]
    all_candidate_features: List[str]
    tunable_parameters: Dict[str, Dict[str, Any]]


class TrainRequest(BaseModel):
    model_type: str = Field(..., description="One of: 'RandomForest', 'LightGBM', 'KerasMLP'")
    model_name: Optional[str] = Field(None, description="Unique model identifier")
    features: Optional[List[str]] = Field(None, description="Subset of candidate features (defaults to all)")
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class TrainResponse(BaseModel):
    model_name: str
    model_type: str
    features: List[str]
    hyperparameters: Dict[str, Any]
    training_time_sec: float
    metrics: Dict[str, Dict[str, float]] # train, val, test
    baseline_metrics: Dict[str, Dict[str, float]]
    feature_importances: Optional[Dict[str, float]] = None
    learning_curve: Optional[Dict[str, List[float]]] = None


class ModelDetailResponse(BaseModel):
    model_name: str
    model_type: str
    features: List[str]
    hyperparameters: Dict[str, Any]
    training_time_sec: float
    created_at: str
    metrics: Dict[str, Dict[str, float]]
    baseline_metrics: Dict[str, Dict[str, float]]
    feature_importances: Optional[Dict[str, float]] = None
    learning_curve: Optional[Dict[str, List[float]]] = None


class PredictRecord(BaseModel):
    Store: int
    Date: str
    Open: int = 1
    Promo: int = 0
    SchoolHoliday: int = 0
    StateHoliday: str = "0"
    StoreType: Optional[str] = None
    Assortment: Optional[str] = None
    CompetitionDistance: Optional[float] = None


class PredictBatchRequest(BaseModel):
    model_name: str
    records: List[PredictRecord]


class SinglePredictRequest(BaseModel):
    model_name: str
    record: PredictRecord


class PredictionItem(BaseModel):
    Store: int
    Date: str
    predicted_sales: float
    predicted_log_sales: float
    is_open: int


class PredictResponse(BaseModel):
    model_name: str
    predictions: List[PredictionItem]


class DayOverride(BaseModel):
    Date: str
    Store: Optional[int] = None
    Open: Optional[int] = None
    Promo: Optional[int] = None
    SchoolHoliday: Optional[int] = None
    StateHoliday: Optional[str] = None


class ForecastRequest(BaseModel):
    model_name: str
    stores: List[int] = Field(default_factory=lambda: [1])
    start_date: str
    end_date: str
    promo_switch: Optional[int] = Field(None, description="Override promo flag for all days: 1 (on) or 0 (off)")
    scenario_overrides: Optional[List[DayOverride]] = Field(default_factory=list)


class ForecastItem(BaseModel):
    Store: int
    Date: str
    predicted_sales: float
    actual_sales: Optional[float] = None
    is_open: int
    promo: int


class ForecastResponse(BaseModel):
    model_name: str
    forecasts: List[ForecastItem]
    per_store_metrics: Optional[Dict[str, Dict[str, float]]] = None
    aggregate_metrics: Optional[Dict[str, float]] = None
    has_actuals: bool = False

