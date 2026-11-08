"""
Data loading and preprocessing utilities for Rossmann Store Sales forecasting.
Strictly adheres to:
- StateHoliday read as string ('0', 'a', 'b', 'c').
- Customers is target leakage (used in EDA only, never as a feature).
- Train on Open == 1 only; Open == 0 forecasts 0.
- logMonthSales recomputed strictly from training period data to avoid data leakage.
"""

import os
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd

# Default data path relative to this file
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
DATA_DIR = os.getenv("DATA_DIR", DEFAULT_DATA_DIR)

CANDIDATE_NUMERICAL = ["Promo", "SchoolHoliday", "logMonthSales", "CompetitionDistance"]
CANDIDATE_CATEGORICAL = ["Store", "StateHoliday", "StoreType", "Assortment", "weekday", "monthday", "month"]
ALL_CANDIDATE_FEATURES = CANDIDATE_NUMERICAL + CANDIDATE_CATEGORICAL

_DATA_CACHE: Dict[str, pd.DataFrame] = {}
_STORE_MONTH_SALES_MAP: Dict[Tuple[int, int], float] = {}
_BASELINE_MAP: Dict[Tuple[int, int], float] = {}


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Loads raw train and test dataframes with caching and correct dtypes."""
    global _DATA_CACHE
    if "train" in _DATA_CACHE and "test" in _DATA_CACHE:
        return _DATA_CACHE["train"].copy(), _DATA_CACHE["test"].copy()

    train_path = os.path.join(DATA_DIR, "rossm_train_df.csv")
    test_path = os.path.join(DATA_DIR, "rossm_test_df.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(f"Data files not found in {DATA_DIR}. Expected rossm_train_df.csv and rossm_test_df.csv.")

    train_df = pd.read_csv(train_path, dtype={"StateHoliday": str})
    test_df = pd.read_csv(test_path, dtype={"StateHoliday": str})

    train_df["StateHoliday"] = train_df["StateHoliday"].fillna("0").astype(str)
    test_df["StateHoliday"] = test_df["StateHoliday"].fillna("0").astype(str)

    _DATA_CACHE["train"] = train_df
    _DATA_CACHE["test"] = test_df
    return train_df.copy(), test_df.copy()


def compute_clean_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    """
    Loads data, splits 2015 into validation (< 2015-04-01) and test (>= 2015-04-01),
    and strictly recomputes logMonthSales from train only, preventing leakage.
    Also computes the naive baseline lookup (Store, weekday) -> mean Sales.
    """
    global _STORE_MONTH_SALES_MAP, _BASELINE_MAP
    train_df, test_df = load_raw_data()

    # Recompute logMonthSales strictly from train open days
    train_open = train_df[train_df["Open"] == 1]
    store_month_mean = train_open.groupby(["Store", "month"])["Sales"].mean().to_dict()
    global_month_mean = train_open.groupby("month")["Sales"].mean().to_dict()
    overall_mean = train_open["Sales"].mean()

    # Map function to compute clean logMonthSales
    def get_clean_log_month_sales(store: int, month: int) -> float:
        val = store_month_mean.get((store, month))
        if val is None or np.isnan(val):
            val = global_month_mean.get(month, overall_mean)
        return float(np.log1p(val))

    for df in (train_df, test_df):
        df["logMonthSales"] = [
            get_clean_log_month_sales(s, m) for s, m in zip(df["Store"], df["month"])
        ]

    # Baseline map: mean training sales for (Store, weekday)
    store_weekday_mean = train_open.groupby(["Store", "weekday"])["Sales"].mean().to_dict()
    global_weekday_mean = train_open.groupby("weekday")["Sales"].mean().to_dict()

    _STORE_MONTH_SALES_MAP = store_month_mean
    _BASELINE_MAP = store_weekday_mean

    # Split test_df into validation (< 2015-04-01) and test (>= 2015-04-01)
    val_df = test_df[test_df["Date"] < "2015-04-01"].copy()
    test_eval_df = test_df[test_df["Date"] >= "2015-04-01"].copy()

    return train_df, val_df, test_eval_df, store_month_mean, store_weekday_mean


def get_naive_baseline_prediction(df: pd.DataFrame, baseline_map: Dict[Tuple[int, int], float] = None) -> np.ndarray:
    """Predicts naive baseline (store's mean training sales for that weekday). Predicts 0 if Open==0."""
    if baseline_map is None:
        global _BASELINE_MAP
        baseline_map = _BASELINE_MAP

    stores = df["Store"].values
    weekdays = df["weekday"].values
    opens = df["Open"].values if "Open" in df.columns else np.ones(len(df))
    preds = [
        0.0 if op == 0 else float(baseline_map.get((int(s), int(w)), 5000.0))
        for s, w, op in zip(stores, weekdays, opens)
    ]
    return np.array(preds, dtype=float)


def get_store_metadata() -> pd.DataFrame:
    """Returns store-level static metadata (StoreType, Assortment, CompetitionDistance)."""
    train_df, _ = load_raw_data()
    return train_df[["Store", "StoreType", "Assortment", "CompetitionDistance"]].drop_duplicates().sort_values("Store").reset_index(drop=True)
