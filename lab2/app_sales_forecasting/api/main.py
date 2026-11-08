"""
FastAPI application for the Sales Forecasting microservice.
"""

import os
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    HealthResponse,
    DataInfoResponse,
    TrainRequest,
    TrainResponse,
    ModelDetailResponse,
    SinglePredictRequest,
    PredictBatchRequest,
    PredictResponse,
    PredictionItem,
    ForecastRequest,
    ForecastResponse
)
from .service import ForecastingService
from .data_loader import (
    load_raw_data,
    CANDIDATE_NUMERICAL,
    CANDIDATE_CATEGORICAL,
    ALL_CANDIDATE_FEATURES
)

# Global service instance
service: Optional[ForecastingService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    service = ForecastingService()
    yield
    # Cleanup if needed


app = FastAPI(
    title="Rossmann Store Sales Forecasting API",
    description="Microservice providing daily sales forecasting, model training, and scenario analysis.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def get_health():
    """Returns service health status and currently available trained models."""
    models = list(service.models_registry.keys()) if service else []
    return HealthResponse(
        status="ok",
        service="sales-forecasting-api",
        available_models=models
    )


@app.get("/data/info", response_model=DataInfoResponse, tags=["Data"])
def get_data_info():
    """Returns metadata about the Rossmann dataset: stores, date ranges, and tunable model parameters."""
    train_df, test_df = load_raw_data()
    store_ids = sorted(train_df["Store"].unique().tolist())
    val_df = test_df[test_df["Date"] < "2015-04-01"]
    test_eval_df = test_df[test_df["Date"] >= "2015-04-01"]

    tunable_params = {
        "RandomForest": {
            "n_estimators": {"type": "int", "default": 100, "description": "Number of trees"},
            "max_depth": {"type": "int", "default": None, "description": "Max tree depth"},
            "min_samples_leaf": {"type": "int", "default": 1, "description": "Min leaf samples"}
        },
        "LightGBM": {
            "n_estimators": {"type": "int", "default": 1000, "description": "Max boosting rounds"},
            "learning_rate": {"type": "float", "default": 0.1, "description": "Step size shrinkage"},
            "num_leaves": {"type": "int", "default": 100, "description": "Max leaves per tree"},
            "max_depth": {"type": "int", "default": 5, "description": "Max depth"},
            "colsample_bytree": {"type": "float", "default": 0.7, "description": "Feature subsample ratio"}
        },
        "KerasMLP": {
            "hidden_layer_sizes": {"type": "list[int]", "default": [512, 256], "description": "Dense layer dimensions"},
            "dropout": {"type": "float", "default": 0.1, "description": "Dropout rate"},
            "learning_rate": {"type": "float", "default": 1e-4, "description": "Adam learning rate"},
            "epochs": {"type": "int", "default": 20, "description": "Training epochs"},
            "batch_size": {"type": "int", "default": 512, "description": "Mini-batch size"}
        }
    }

    return DataInfoResponse(
        store_ids=store_ids,
        total_stores=len(store_ids),
        train_date_range={"start": str(train_df["Date"].min()), "end": str(train_df["Date"].max())},
        val_date_range={"start": str(val_df["Date"].min()), "end": str(val_df["Date"].max())},
        test_date_range={"start": str(test_eval_df["Date"].min()), "end": str(test_eval_df["Date"].max())},
        candidate_numerical_features=CANDIDATE_NUMERICAL,
        candidate_categorical_features=CANDIDATE_CATEGORICAL,
        all_candidate_features=ALL_CANDIDATE_FEATURES,
        tunable_parameters=tunable_params
    )


@app.post("/train", response_model=TrainResponse, tags=["Training"])
def train_model_endpoint(req: TrainRequest):
    """
    Trains a model (RandomForest, LightGBM, or KerasMLP) on the training set,
    evaluating on train, validation (< 2015-04-01), and test (>= 2015-04-01).
    """
    if req.model_type not in ["RandomForest", "LightGBM", "KerasMLP"]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported model type '{req.model_type}'. Must be RandomForest, LightGBM, or KerasMLP."
        )

    try:
        meta = service.train_model(
            model_type=req.model_type,
            model_name=req.model_name,
            features=req.features,
            hyperparameters=req.hyperparameters
        )
        return TrainResponse(
            model_name=meta["model_name"],
            model_type=meta["model_type"],
            features=meta["features"],
            hyperparameters=meta["hyperparameters"],
            training_time_sec=meta["training_time_sec"],
            metrics=meta["metrics"],
            baseline_metrics=meta["baseline_metrics"],
            feature_importances=meta.get("feature_importances"),
            learning_curve=meta.get("learning_curve")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/models", response_model=List[ModelDetailResponse], tags=["Models"])
def list_models():
    """Lists all saved models with their training metrics and feature sets."""
    if not service.models_registry:
        return []
    res = []
    for m in service.list_models():
        res.append(ModelDetailResponse(**m))
    return res


@app.get("/models/{name}", response_model=ModelDetailResponse, tags=["Models"])
def get_model(name: str):
    """Retrieves metadata and metrics for a specific model."""
    if name not in service.models_registry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model '{name}' not found.")
    meta = service.models_registry[name]["meta"]
    return ModelDetailResponse(**meta)


@app.delete("/models/{name}", tags=["Models"])
def delete_model(name: str):
    """Deletes a model and removes its files from disk."""
    if name not in service.models_registry:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model '{name}' not found.")
    deleted = service.delete_model(name)
    return {"status": "deleted", "model_name": name, "success": deleted}


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_single(req: SinglePredictRequest):
    """Generates daily sales forecast for a single store-day."""
    if not service.models_registry:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No trained model available yet. Please train a model first via POST /train."
        )
    if req.model_name not in service.models_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{req.model_name}' not found."
        )

    preds = service.predict_records(req.model_name, [req.record.model_dump()])
    return PredictResponse(
        model_name=req.model_name,
        predictions=[PredictionItem(**p) for p in preds]
    )


@app.post("/predict/batch", response_model=PredictResponse, tags=["Inference"])
def predict_batch(req: PredictBatchRequest):
    """Generates daily sales forecasts for a batch of store-days."""
    if not service.models_registry:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No trained model available yet. Please train a model first via POST /train."
        )
    if req.model_name not in service.models_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{req.model_name}' not found."
        )

    records = [r.model_dump() for r in req.records]
    preds = service.predict_records(req.model_name, records)
    return PredictResponse(
        model_name=req.model_name,
        predictions=[PredictionItem(**p) for p in preds]
    )


@app.post("/forecast", response_model=ForecastResponse, tags=["Inference"])
def forecast(req: ForecastRequest):
    """
    Generates forecasts for multiple stores across a date range.
    Supports holdout evaluation against actual test sales and scenario overrides.
    """
    if not service.models_registry:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No trained model available yet. Please train a model first via POST /train."
        )
    if req.model_name not in service.models_registry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{req.model_name}' not found."
        )

    overrides = [ov.model_dump() for ov in req.scenario_overrides] if req.scenario_overrides else None
    result = service.generate_forecast(
        model_name=req.model_name,
        stores=req.stores,
        start_date=req.start_date,
        end_date=req.end_date,
        promo_switch=req.promo_switch,
        scenario_overrides=overrides
    )
    return ForecastResponse(**result)

