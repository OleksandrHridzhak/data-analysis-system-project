"""
Tests ensuring strict adherence to leakage prevention rules and domain constraints:
1. Customers is NEVER a candidate feature (leakage).
2. StateHoliday is always treated as string ('0', 'a', etc.).
3. logMonthSales is computed strictly from training period, never from full/test period.
4. Closed days (Open == 0) strictly predict 0.0 sales.
"""

import pytest
import numpy as np
import pandas as pd
from api.data_loader import (
    load_raw_data,
    compute_clean_datasets,
    ALL_CANDIDATE_FEATURES,
    CANDIDATE_NUMERICAL,
    CANDIDATE_CATEGORICAL
)
from api.service import ForecastingService


def test_customers_not_in_candidate_features():
    """Customers is known only post-hoc and must never be a feature."""
    assert "Customers" not in ALL_CANDIDATE_FEATURES
    assert "Customers" not in CANDIDATE_NUMERICAL
    assert "Customers" not in CANDIDATE_CATEGORICAL


def test_state_holiday_is_string():
    """StateHoliday mixes 0 with letters and must be loaded as string."""
    train_df, test_df = load_raw_data()
    assert train_df["StateHoliday"].dtype == object or pd.api.types.is_string_dtype(train_df["StateHoliday"])
    assert test_df["StateHoliday"].dtype == object or pd.api.types.is_string_dtype(test_df["StateHoliday"])
    assert "0" in train_df["StateHoliday"].values


def test_log_month_sales_recomputed_from_train_only():
    """logMonthSales must not leak test data."""
    train_df, val_df, test_df, store_month_mean, _ = compute_clean_datasets()
    
    # Store 1, month 1 mean computed on train open days
    train_open = train_df[train_df["Open"] == 1]
    expected_s1_m1 = train_open[(train_open["Store"] == 1) & (train_open["month"] == 1)]["Sales"].mean()
    
    # Verify the computed map matches
    assert np.isclose(store_month_mean[(1, 1)], expected_s1_m1, rtol=1e-5)


def test_closed_days_forecast_zero(tmp_path):
    """When Open == 0, forecast must be exactly 0.0."""
    service = ForecastingService(models_dir=str(tmp_path))
    # Quick train
    train_res = service.train_model(
        model_type="RandomForest",
        model_name="test_rf_zero",
        features=["Promo", "Store", "weekday"],
        hyperparameters={"n_estimators": 5, "max_depth": 3, "random_state": 42}
    )
    
    closed_record = {
        "Store": 1,
        "Date": "2015-05-03", # Sunday
        "Open": 0,
        "Promo": 0,
        "SchoolHoliday": 0,
        "StateHoliday": "0"
    }
    preds = service.predict_records("test_rf_zero", [closed_record])
    assert len(preds) == 1
    assert preds[0]["predicted_sales"] == 0.0
    assert preds[0]["predicted_log_sales"] == 0.0
    assert preds[0]["is_open"] == 0
