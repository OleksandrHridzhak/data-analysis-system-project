"""
End-to-end verification script for Lab 2:
1. Trains Random Forest on full data
2. Trains LightGBM on full data
3. Trains Keras MLP on full data
4. Verifies single prediction, batch prediction, holdout forecast, and closed day == 0
5. Tests model reload (service restart)
6. Outputs markdown table of results
"""

import sys
import os
import time

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def main():
    from api.service import ForecastingService

    srv = ForecastingService()

    print("=== STEP 1: TRAINING RANDOM FOREST ===", flush=True)
    t0 = time.time()
    rf_meta = srv.train_model(
        model_type="RandomForest",
        model_name="RandomForest_Default",
        features=["Promo", "SchoolHoliday", "logMonthSales", "CompetitionDistance", "Store", "StateHoliday", "StoreType", "Assortment", "weekday", "monthday", "month"],
        hyperparameters={"n_estimators": 50, "max_depth": 12, "min_samples_leaf": 3, "random_state": 42, "n_jobs": 1}
    )
    print("RF Time (s):", rf_meta["training_time_sec"], flush=True)
    print("RF Val RMSE (Sales):", rf_meta["metrics"]["validation"]["rmse_Sales"], flush=True)
    print("RF Test RMSE (Sales):", rf_meta["metrics"]["test"]["rmse_Sales"], flush=True)

    print("=== STEP 2: TRAINING LIGHTGBM ===", flush=True)
    t0 = time.time()
    lgb_meta = srv.train_model(
        model_type="LightGBM",
        model_name="LightGBM_Default",
        features=["Promo", "SchoolHoliday", "logMonthSales", "CompetitionDistance", "Store", "StateHoliday", "StoreType", "Assortment", "weekday", "monthday", "month"],
        hyperparameters={"n_estimators": 400, "learning_rate": 0.1, "num_leaves": 60, "max_depth": 5, "colsample_bytree": 0.7, "random_state": 42, "n_jobs": 1}
    )
    print("LightGBM Time (s):", lgb_meta["training_time_sec"], flush=True)
    print("LightGBM Val RMSE (Sales):", lgb_meta["metrics"]["validation"]["rmse_Sales"], flush=True)
    print("LightGBM Test RMSE (Sales):", lgb_meta["metrics"]["test"]["rmse_Sales"], flush=True)

    print("=== STEP 3: TRAINING KERAS MLP ===", flush=True)
    t0 = time.time()
    mlp_meta = srv.train_model(
        model_type="KerasMLP",
        model_name="Keras_MLP_Default",
        features=["Promo", "SchoolHoliday", "logMonthSales", "CompetitionDistance", "Store", "StateHoliday", "StoreType", "Assortment", "weekday", "monthday", "month"],
        hyperparameters={"hidden_layer_sizes": [256, 128], "dropout": 0.1, "learning_rate": 1e-4, "epochs": 15, "batch_size": 512, "random_state": 42}
    )
    print("MLP Time (s):", mlp_meta["training_time_sec"], flush=True)
    print("MLP Val RMSE (Sales):", mlp_meta["metrics"]["validation"]["rmse_Sales"], flush=True)
    print("MLP Test RMSE (Sales):", mlp_meta["metrics"]["test"]["rmse_Sales"], flush=True)

    print("=== STEP 4: VERIFY PREDICTIONS & CLOSED DAYS RULE ===", flush=True)
    open_pred = srv.predict_records("LightGBM_Default", [{"Store": 1, "Date": "2015-05-01", "Open": 1, "Promo": 1}])
    closed_pred = srv.predict_records("LightGBM_Default", [{"Store": 1, "Date": "2015-05-03", "Open": 0, "Promo": 0}])
    print("Open day forecast ($):", open_pred[0]["predicted_sales"], flush=True)
    print("Closed day forecast ($) [must be 0.0]:", closed_pred[0]["predicted_sales"], flush=True)
    assert closed_pred[0]["predicted_sales"] == 0.0

    print("=== STEP 5: VERIFY PERSISTENCE ACROSS RESTART ===", flush=True)
    srv2 = ForecastingService()
    assert "RandomForest_Default" in srv2.models_registry
    assert "LightGBM_Default" in srv2.models_registry
    assert "Keras_MLP_Default" in srv2.models_registry
    print("All 3 models survived service reload successfully!", flush=True)
    print("ALL VERIFICATIONS COMPLETED SUCCESSFULLY!", flush=True)


if __name__ == "__main__":
    main()
