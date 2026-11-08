import pytest
import pandas as pd
import numpy as np
from src.pipeline import build_pipeline
from src.data_generator import generate_real_estate_data
from src.config import TARGET_COL


def test_build_pipeline_and_predict():
    df = generate_real_estate_data(n_samples=150, random_seed=42)
    
    num_cols = ["living_area_sqft", "lot_size_sqft", "bedrooms", "bathrooms"]
    cat_cols = ["city", "property_type", "property_condition"]
    
    pipeline = build_pipeline(
        numerical_features=num_cols,
        categorical_features=cat_cols,
        n_estimators=10,
        max_depth=5,
        random_state=42
    )
    
    X = df[num_cols + cat_cols]
    y = df[TARGET_COL]
    
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    
    assert len(preds) == len(df)
    assert np.issubdtype(preds.dtype, np.floating)
    assert not np.isnan(preds).any()


def test_pipeline_missing_values_handling():
    df = generate_real_estate_data(n_samples=100, random_seed=42)
    
    num_cols = ["living_area_sqft", "basement_sqft"]
    cat_cols = ["city", "roof_type"]
    
    # Inject extra missing values
    df_missing = df[num_cols + cat_cols].copy()
    df_missing.loc[0:5, "basement_sqft"] = np.nan
    df_missing.loc[5:10, "roof_type"] = np.nan
    
    pipeline = build_pipeline(
        numerical_features=num_cols,
        categorical_features=cat_cols,
        n_estimators=10,
        max_depth=4,
        random_state=42
    )
    
    pipeline.fit(df_missing, df[TARGET_COL])
    preds = pipeline.predict(df_missing)
    
    assert len(preds) == len(df_missing)
    assert not np.isnan(preds).any()

