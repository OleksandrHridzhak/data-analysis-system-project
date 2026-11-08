import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_generator import generate_real_estate_data
from src.model_manager import train_and_evaluate, save_model_bundle, load_model_bundle
from src.analytics import compute_categorical_aggregation, compute_numerical_sensitivity
from src.config import TARGET_COL


def test_model_training_and_metrics():
    df = generate_real_estate_data(n_samples=250, random_seed=42)
    
    num_cols = ["living_area_sqft", "lot_size_sqft", "school_rating"]
    cat_cols = ["city", "property_type"]
    
    bundle = train_and_evaluate(
        df=df,
        numerical_features=num_cols,
        categorical_features=cat_cols,
        test_size=0.25,
        n_estimators=15,
        max_depth=6,
        random_state=42
    )
    
    assert "pipeline" in bundle
    assert "test_metrics" in bundle
    assert "train_metrics" in bundle
    
    metrics = bundle["test_metrics"]
    assert metrics["MAE"] > 0
    assert metrics["RMSE"] > 0
    assert -1.0 <= metrics["R2"] <= 1.0
    assert metrics["MAPE"] > 0
    
    assert len(bundle["df_importance"]) > 0


def test_model_persistence(tmp_path):
    df = generate_real_estate_data(n_samples=150, random_seed=42)
    num_cols = ["living_area_sqft", "school_rating"]
    cat_cols = ["city"]
    
    bundle = train_and_evaluate(
        df=df,
        numerical_features=num_cols,
        categorical_features=cat_cols,
        n_estimators=10,
        max_depth=4,
        random_state=42
    )
    
    save_file = tmp_path / "test_model.joblib"
    save_model_bundle(bundle, filepath=save_file)
    assert save_file.exists()
    
    loaded_bundle = load_model_bundle(filepath=save_file)
    assert loaded_bundle is not None
    assert loaded_bundle["all_features"] == bundle["all_features"]
    
    # Test identical predictions
    sample_X = df[bundle["all_features"]].head(10)
    orig_preds = bundle["pipeline"].predict(sample_X)
    loaded_preds = loaded_bundle["pipeline"].predict(sample_X)
    np.testing.assert_allclose(orig_preds, loaded_preds)


def test_categorical_aggregation_and_sensitivity():
    df = generate_real_estate_data(n_samples=150, random_seed=42)
    num_cols = ["living_area_sqft", "lot_size_sqft"]
    cat_cols = ["city", "property_condition"]
    
    bundle = train_and_evaluate(
        df=df,
        numerical_features=num_cols,
        categorical_features=cat_cols,
        n_estimators=10,
        max_depth=4,
        random_state=42
    )
    
    # Categorical aggregation test
    eval_df = bundle["eval_sample"]
    agg_df = compute_categorical_aggregation(eval_df, group_col="city")
    assert "Mean Actual ($)" in agg_df.columns
    assert "Mean Predicted ($)" in agg_df.columns
    assert "MAE ($)" in agg_df.columns
    assert len(agg_df) > 0
    
    # Sensitivity test
    base_prop = eval_df.iloc[0].to_dict()
    sens_df = compute_numerical_sensitivity(
        pipeline=bundle["pipeline"],
        base_property=base_prop,
        feature_name="living_area_sqft",
        min_val=1000,
        max_val=4000,
        steps=10
    )
    assert len(sens_df) == 10
    assert "Predicted_Price" in sens_df.columns

