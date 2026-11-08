import pytest
import pandas as pd
import numpy as np
from src.data_generator import generate_real_estate_data
from src.config import TARGET_COL, CITIES_PROFILE


def test_data_generation_shape_and_columns():
    n_samples = 250
    df = generate_real_estate_data(n_samples=n_samples, random_seed=42)
    
    assert len(df) == n_samples
    assert TARGET_COL in df.columns
    assert "city" in df.columns
    assert "living_area_sqft" in df.columns
    assert "property_type" in df.columns
    assert df.shape[1] >= 35


def test_target_price_validity():
    df = generate_real_estate_data(n_samples=300, random_seed=123)
    prices = df[TARGET_COL]
    
    assert (prices > 0).all(), "All prices must be strictly positive"
    assert prices.min() >= 50000, "Prices should be above reasonable minimum"
    assert not prices.isnull().any(), "Price should not contain missing values"


def test_cities_representation():
    df = generate_real_estate_data(n_samples=500, random_seed=42)
    expected_cities = set(CITIES_PROFILE.keys())
    present_cities = set(df["city"].unique())
    
    assert present_cities.issubset(expected_cities)
    assert len(present_cities) == len(expected_cities)


def test_no_target_leakage():
    df = generate_real_estate_data(n_samples=100, random_seed=99)
    # Ensure there are no derived leakage columns like price_per_sqft
    assert "price_per_sqft" not in df.columns
    assert "price_log" not in df.columns

