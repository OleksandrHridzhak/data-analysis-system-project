import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATA_FILE = DATA_DIR / "properties.csv"
MODEL_FILE = MODELS_DIR / "trained_model.joblib"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Target variable
TARGET_COL = "price"

# Cities and their realistic geographic and economic baseline profiles
CITIES_PROFILE = {
    "Austin": {
        "state": "TX", "lat": 30.2672, "lon": -97.7431, "base_price_sqft": 310.0,
        "income_mean": 86000, "income_std": 18000, "tax_rate": 0.0195
    },
    "Seattle": {
        "state": "WA", "lat": 47.6062, "lon": -122.3321, "base_price_sqft": 520.0,
        "income_mean": 105000, "income_std": 22000, "tax_rate": 0.0098
    },
    "Chicago": {
        "state": "IL", "lat": 41.8781, "lon": -87.6298, "base_price_sqft": 270.0,
        "income_mean": 75000, "income_std": 19000, "tax_rate": 0.0210
    },
    "Miami": {
        "state": "FL", "lat": 25.7617, "lon": -80.1918, "base_price_sqft": 480.0,
        "income_mean": 68000, "income_std": 21000, "tax_rate": 0.0102
    },
    "Denver": {
        "state": "CO", "lat": 39.7392, "lon": -104.9903, "base_price_sqft": 390.0,
        "income_mean": 89000, "income_std": 17000, "tax_rate": 0.0065
    },
    "New York": {
        "state": "NY", "lat": 40.7128, "lon": -74.0060, "base_price_sqft": 680.0,
        "income_mean": 92000, "income_std": 32000, "tax_rate": 0.0145
    },
    "San Francisco": {
        "state": "CA", "lat": 37.7749, "lon": -122.4194, "base_price_sqft": 790.0,
        "income_mean": 126000, "income_std": 35000, "tax_rate": 0.0076
    },
    "Atlanta": {
        "state": "GA", "lat": 33.7490, "lon": -84.3880, "base_price_sqft": 240.0,
        "income_mean": 77000, "income_std": 18000, "tax_rate": 0.0125
    },
    "Boston": {
        "state": "MA", "lat": 42.3601, "lon": -71.0589, "base_price_sqft": 590.0,
        "income_mean": 98000, "income_std": 24000, "tax_rate": 0.0118
    },
    "Phoenix": {
        "state": "AZ", "lat": 33.4484, "lon": -112.0740, "base_price_sqft": 260.0,
        "income_mean": 72000, "income_std": 16000, "tax_rate": 0.0062
    },
}

PROPERTY_TYPES = ["Single Family", "Condo", "Townhouse", "Multi-Family"]
PROPERTY_CONDITIONS = ["Poor", "Fair", "Good", "Excellent", "Luxury"]
ROOF_TYPES = ["Asphalt Shingle", "Tile", "Metal", "Slate", "Flat"]
HEATING_TYPES = ["Central Gas", "Electric Heat Pump", "Forced Air", "Radiant", "Baseboard"]

# Numerical and categorical feature lists
NUMERICAL_FEATURES = [
    "latitude",
    "longitude",
    "distance_to_downtown_miles",
    "living_area_sqft",
    "lot_size_sqft",
    "bedrooms",
    "bathrooms",
    "stories",
    "rooms_total",
    "year_built",
    "property_age",
    "renovation_age",
    "garage_spaces",
    "basement_sqft",
    "school_rating",
    "safety_score",
    "walk_score",
    "transit_score",
    "park_proximity_miles",
    "neighborhood_median_income",
    "local_property_tax_rate",
    "hoa_fee_monthly",
    "days_on_market",
    "mortgage_rate_at_listing",
    "energy_efficiency_rating"
]

CATEGORICAL_FEATURES = [
    "city",
    "state",
    "property_type",
    "property_condition",
    "roof_type",
    "heating_type",
    "has_garage",
    "has_pool",
    "has_basement",
    "has_garden",
    "has_fireplace",
    "has_central_ac",
    "has_solar_panels",
    "has_been_renovated"
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

