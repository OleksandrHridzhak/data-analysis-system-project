"""
Service layer orchestrating data, feature preprocessing, model lifecycle, inference and forecasting.
"""

import os
import json
import time
import datetime
from typing import Dict, Any, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from .data_loader import (
    compute_clean_datasets,
    load_raw_data,
    get_naive_baseline_prediction,
    get_store_metadata,
    CANDIDATE_NUMERICAL,
    CANDIDATE_CATEGORICAL,
    ALL_CANDIDATE_FEATURES
)
from .features import FeaturePreprocessor
from .models import (
    BaseModel,
    RandomForestModel,
    LightGBMModel,
    KerasMLPModel,
    calculate_metrics
)

SAVED_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_models"))


class ForecastingService:
    """Manages training, model persistence, inference and scenario forecasting."""

    def __init__(self, models_dir: str = SAVED_MODELS_DIR):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.models_registry: Dict[str, Dict[str, Any]] = {}
        self.store_meta_df = get_store_metadata()
        self.reload_saved_models()

    def reload_saved_models(self) -> None:
        """Discovers and reloads all persisted models from disk on startup."""
        self.models_registry.clear()
        if not os.path.exists(self.models_dir):
            return

        for fname in os.listdir(self.models_dir):
            if fname.endswith("_meta.json"):
                meta_path = os.path.join(self.models_dir, fname)
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    model_name = meta["model_name"]
                    model_type = meta["model_type"]

                    # Load preprocessor
                    prep_path = os.path.join(self.models_dir, f"{model_name}_preprocessor.joblib")
                    preprocessor = joblib.load(prep_path)

                    # Load model
                    if model_type == "RandomForest":
                        model = RandomForestModel.load(self.models_dir, model_name)
                    elif model_type == "LightGBM":
                        model = LightGBMModel.load(self.models_dir, model_name)
                    elif model_type == "KerasMLP":
                        model = KerasMLPModel.load(self.models_dir, model_name)
                    else:
                        continue

                    self.models_registry[model_name] = {
                        "model": model,
                        "preprocessor": preprocessor,
                        "meta": meta
                    }
                except Exception as e:
                    print(f"Error loading model from {fname}: {e}")

    def list_models(self) -> List[Dict[str, Any]]:
        """Returns summary list of all registered models."""
        return [entry["meta"] for entry in self.models_registry.values()]

    def get_model_entry(self, model_name: str) -> Dict[str, Any]:
        """Returns model bundle or raises KeyError."""
        if model_name not in self.models_registry:
            raise KeyError(f"Model '{model_name}' not found.")
        return self.models_registry[model_name]

    def delete_model(self, model_name: str) -> bool:
        """Deletes a model and all its on-disk artifacts."""
        if model_name not in self.models_registry:
            return False

        del self.models_registry[model_name]
        # Delete files matching model_name
        for fname in os.listdir(self.models_dir):
            if fname.startswith(f"{model_name}_"):
                try:
                    os.remove(os.path.join(self.models_dir, fname))
                except OSError:
                    pass
        return True

    def train_model(
        self,
        model_type: str,
        model_name: Optional[str] = None,
        features: Optional[List[str]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Trains model following strict methodology:
        - Recomputes logMonthSales from train only (no leakage)
        - Trains strictly on open days (Open == 1)
        - Computes baseline & model metrics on train, val (< 2015-04-01), and test (>= 2015-04-01)
        """
        t0 = time.time()
        if hyperparameters is None:
            hyperparameters = {}

        if not model_name:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"{model_type}_{ts}"

        if not features:
            features = list(ALL_CANDIDATE_FEATURES)

        # 1. Load clean datasets
        train_df, val_df, test_df, store_month_mean, baseline_map = compute_clean_datasets()

        # 2. Filter open days for training & metric reporting
        train_open = train_df[train_df["Open"] == 1].copy()
        val_open = val_df[val_df["Open"] == 1].copy()
        test_open = test_df[test_df["Open"] == 1].copy()

        # 3. Fit Preprocessor strictly on training open days
        preprocessor = FeaturePreprocessor(features)
        X_train_trans = preprocessor.fit_transform(train_open[features])
        X_val_trans = preprocessor.transform(val_open[features])
        X_test_trans = preprocessor.transform(test_open[features])

        y_train = train_open["logSales"].values
        y_val = val_open["logSales"].values
        y_test = test_open["logSales"].values

        # 4. Instantiate Model
        if model_type == "RandomForest":
            model = RandomForestModel(model_name, hyperparameters)
        elif model_type == "LightGBM":
            model = LightGBMModel(model_name, hyperparameters)
        elif model_type == "KerasMLP":
            model = KerasMLPModel(model_name, hyperparameters)
        else:
            raise ValueError(f"Unsupported model_type: {model_type}. Must be RandomForest, LightGBM, or KerasMLP.")

        # 5. Fit Model
        model.fit(X_train_trans, y_train, X_val=X_val_trans, y_val=y_val)
        training_time = round(time.time() - t0, 2)

        # 6. Predict on open days
        pred_train_log = model.predict(X_train_trans)
        pred_val_log = model.predict(X_val_trans)
        pred_test_log = model.predict(X_test_trans)

        # 7. Model Metrics
        metrics = {
            "train": calculate_metrics(y_train, pred_train_log),
            "validation": calculate_metrics(y_val, pred_val_log),
            "test": calculate_metrics(y_test, pred_test_log)
        }

        # 8. Baseline Metrics
        base_train_sales = get_naive_baseline_prediction(train_open, baseline_map)
        base_val_sales = get_naive_baseline_prediction(val_open, baseline_map)
        base_test_sales = get_naive_baseline_prediction(test_open, baseline_map)

        baseline_metrics = {
            "train": calculate_metrics(y_train, np.log1p(base_train_sales)),
            "validation": calculate_metrics(y_val, np.log1p(base_val_sales)),
            "test": calculate_metrics(y_test, np.log1p(base_test_sales))
        }

        # 9. Feature importances (RF and LightGBM)
        feature_importances = None
        raw_fi = model.get_feature_importance()
        if raw_fi is not None:
            feature_importances = preprocessor.aggregate_feature_importances(raw_fi)

        # 10. Learning curves (LightGBM and KerasMLP)
        learning_curve = model.get_learning_curve()

        # 11. Save model, preprocessor, and metadata
        model.save(self.models_dir)
        joblib.dump(preprocessor, os.path.join(self.models_dir, f"{model_name}_preprocessor.joblib"))

        meta = {
            "model_name": model_name,
            "model_type": model_type,
            "features": features,
            "hyperparameters": hyperparameters,
            "training_time_sec": training_time,
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "feature_importances": feature_importances,
            "learning_curve": learning_curve,
            "store_month_mean": {f"{k[0]}_{k[1]}": v for k, v in store_month_mean.items()}
        }
        with open(os.path.join(self.models_dir, f"{model_name}_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self.models_registry[model_name] = {
            "model": model,
            "preprocessor": preprocessor,
            "meta": meta
        }

        return meta

    def prepare_dataframe_for_inference(self, records: List[Dict[str, Any]], preprocessor: FeaturePreprocessor, store_month_map: Dict[Any, float]) -> pd.DataFrame:
        """Enriches raw input records with calendar features, static store info, and clean logMonthSales."""
        df = pd.DataFrame(records)

        # Fill missing static store info
        if "StoreType" not in df.columns or df["StoreType"].isnull().any():
            df = df.drop(columns=[c for c in ["StoreType", "Assortment", "CompetitionDistance"] if c in df.columns])
            df = df.merge(self.store_meta_df, on="Store", how="left")

        # Parse date and calendar features
        dt_series = pd.to_datetime(df["Date"])
        df["year"] = dt_series.dt.year
        df["month"] = dt_series.dt.month
        df["monthday"] = dt_series.dt.day
        df["weekday"] = dt_series.dt.weekday + 1

        if "StateHoliday" not in df.columns:
            df["StateHoliday"] = "0"
        df["StateHoliday"] = df["StateHoliday"].fillna("0").astype(str)

        if "Promo" not in df.columns:
            df["Promo"] = 0
        if "SchoolHoliday" not in df.columns:
            df["SchoolHoliday"] = 0
        if "Open" not in df.columns:
            df["Open"] = 1

        # Map clean logMonthSales
        log_month_vals = []
        for s, m in zip(df["Store"], df["month"]):
            key_str = f"{s}_{m}"
            mean_val = store_month_map.get(key_str, store_month_map.get((s, m), 5000.0))
            log_month_vals.append(float(np.log1p(mean_val)))
        df["logMonthSales"] = log_month_vals

        return df

    def predict_records(self, model_name: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Inference with Open==0 rule: closed days forecast strictly 0."""
        bundle = self.get_model_entry(model_name)
        model: BaseModel = bundle["model"]
        preprocessor: FeaturePreprocessor = bundle["preprocessor"]
        store_month_map = bundle["meta"].get("store_month_mean", {})

        df = self.prepare_dataframe_for_inference(records, preprocessor, store_month_map)

        predictions = []
        # Transform features
        X_trans = preprocessor.transform(df[preprocessor.selected_features])
        pred_logs = model.predict(X_trans)

        stores = df["Store"].values
        dates = df["Date"].values
        opens = df["Open"].values

        for s, d, op, p_log in zip(stores, dates, opens, pred_logs):
            is_open = int(op)
            if is_open == 0:
                pred_sales = 0.0
                pred_log = 0.0
            else:
                pred_log = float(np.clip(p_log, 0.0, 15.0))
                pred_sales = float(round(np.expm1(pred_log), 2))

            predictions.append({
                "Store": int(s),
                "Date": str(d),
                "predicted_sales": pred_sales,
                "predicted_log_sales": round(pred_log, 4),
                "is_open": is_open
            })

        return predictions

    def generate_forecast(
        self,
        model_name: str,
        stores: List[int],
        start_date: str,
        end_date: str,
        promo_switch: Optional[int] = None,
        scenario_overrides: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generates daily forecasts for specified stores and date range.
        If dates exist in test dataset, evaluates actual vs forecast with metrics.
        """
        bundle = self.get_model_entry(model_name)
        _, raw_test_df = load_raw_data()
        raw_test_df["Date"] = raw_test_df["Date"].astype(str)

        dates = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d").tolist()
        records = []
        override_lookup = {}
        if scenario_overrides:
            for ov in scenario_overrides:
                key = (ov.get("Store"), ov.get("Date"))
                override_lookup[key] = ov
                override_lookup[(None, ov.get("Date"))] = ov

        for dt_str in dates:
            for s_id in stores:
                # Check if in raw_test_df
                match = raw_test_df[(raw_test_df["Store"] == s_id) & (raw_test_df["Date"] == dt_str)]
                if not match.empty:
                    rec = match.iloc[0].to_dict()
                else:
                    # Construct record from dates & static store info
                    rec = {"Store": s_id, "Date": dt_str, "Open": 1, "Promo": 0, "SchoolHoliday": 0, "StateHoliday": "0"}

                # Apply promo_switch if requested
                if promo_switch is not None:
                    rec["Promo"] = int(promo_switch)

                # Apply scenario overrides
                ov = override_lookup.get((s_id, dt_str), override_lookup.get((None, dt_str)))
                if ov:
                    for k in ["Open", "Promo", "SchoolHoliday", "StateHoliday"]:
                        if ov.get(k) is not None:
                            rec[k] = ov[k]

                records.append(rec)

        # Run inference
        preds = self.predict_records(model_name, records)

        # Pair with actuals if available
        forecast_items = []
        actuals_exist = False
        y_true_list = []
        y_pred_list = []
        per_store_data: Dict[int, Dict[str, List[float]]] = {s: {"actual": [], "pred": []} for s in stores}

        for rec, pred in zip(records, preds):
            store = pred["Store"]
            dt = pred["Date"]
            pred_s = pred["predicted_sales"]
            is_op = pred["is_open"]
            promo_val = rec.get("Promo", 0)

            actual_s = None
            if "Sales" in rec and not pd.isna(rec["Sales"]):
                actual_s = float(rec["Sales"])
                actuals_exist = True
                if is_op == 1:
                    y_true_list.append(actual_s)
                    y_pred_list.append(pred_s)
                    per_store_data[store]["actual"].append(actual_s)
                    per_store_data[store]["pred"].append(pred_s)

            forecast_items.append({
                "Store": store,
                "Date": dt,
                "predicted_sales": pred_s,
                "actual_sales": actual_s,
                "is_open": is_op,
                "promo": promo_val
            })

        per_store_metrics = {}
        aggregate_metrics = None
        if actuals_exist and len(y_true_list) > 0:
            y_t = np.array(y_true_list)
            y_p = np.array(y_pred_list)
            aggregate_metrics = calculate_metrics(np.log1p(y_t), np.log1p(y_p))

            for s_id, s_vals in per_store_data.items():
                if len(s_vals["actual"]) > 0:
                    st_act = np.array(s_vals["actual"])
                    st_pred = np.array(s_vals["pred"])
                    per_store_metrics[str(s_id)] = calculate_metrics(np.log1p(st_act), np.log1p(st_pred))

        return {
            "model_name": model_name,
            "forecasts": forecast_items,
            "per_store_metrics": per_store_metrics if actuals_exist else None,
            "aggregate_metrics": aggregate_metrics if actuals_exist else None,
            "has_actuals": actuals_exist
        }
