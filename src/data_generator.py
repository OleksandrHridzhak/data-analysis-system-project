import numpy as np
import pandas as pd
from src.config import (
    CITIES_PROFILE,
    PROPERTY_TYPES,
    PROPERTY_CONDITIONS,
    ROOF_TYPES,
    HEATING_TYPES,
    TARGET_COL,
)


def generate_real_estate_data(n_samples: int = 20000, random_seed: int = 42) -> pd.DataFrame:
    """
    Vectorized generation of realistic synthetic US residential properties data.
    Generates 20,000+ records in ~1 second.
    """
    np.random.seed(random_seed)
    cities = list(CITIES_PROFILE.keys())
    
    # Randomly assign cities
    city_indices = np.random.choice(len(cities), size=n_samples)
    city_arr = np.array(cities)[city_indices]
    
    # Extract city profile arrays
    state_arr = np.array([CITIES_PROFILE[c]["state"] for c in cities])[city_indices]
    base_lat_arr = np.array([CITIES_PROFILE[c]["lat"] for c in cities])[city_indices]
    base_lon_arr = np.array([CITIES_PROFILE[c]["lon"] for c in cities])[city_indices]
    base_price_sqft_arr = np.array([CITIES_PROFILE[c]["base_price_sqft"] for c in cities])[city_indices]
    income_mean_arr = np.array([CITIES_PROFILE[c]["income_mean"] for c in cities])[city_indices]
    income_std_arr = np.array([CITIES_PROFILE[c]["income_std"] for c in cities])[city_indices]
    tax_rate_arr = np.array([CITIES_PROFILE[c]["tax_rate"] for c in cities])[city_indices]
    
    # Coordinates & Distance
    lat_offsets = np.random.normal(0, 0.08, n_samples)
    lon_offsets = np.random.normal(0, 0.08, n_samples)
    latitudes = np.round(base_lat_arr + lat_offsets, 5)
    longitudes = np.round(base_lon_arr + lon_offsets, 5)
    
    dist_miles = np.sqrt((lat_offsets * 69.0) ** 2 + (lon_offsets * 53.0) ** 2)
    dist_miles = np.round(np.clip(dist_miles, 0.2, 35.0), 2)
    
    # Property Types
    prop_type_choices = np.array(PROPERTY_TYPES)
    prop_types = np.random.choice(prop_type_choices, size=n_samples, p=[0.60, 0.22, 0.13, 0.05])
    
    # Living area and Lot size based on property type
    living_sqft = np.zeros(n_samples, dtype=int)
    lot_size_sqft = np.zeros(n_samples, dtype=float)
    stories = np.ones(n_samples, dtype=int)
    
    # Masks
    mask_condo = prop_types == "Condo"
    mask_town = prop_types == "Townhouse"
    mask_multi = prop_types == "Multi-Family"
    mask_single = prop_types == "Single Family"
    
    # Condos
    n_c = np.sum(mask_condo)
    if n_c > 0:
        living_sqft[mask_condo] = np.clip(np.random.normal(1100, 350, n_c), 450, 3200).astype(int)
        lot_size_sqft[mask_condo] = 0.0
        stories[mask_condo] = 1
        
    # Townhouses
    n_t = np.sum(mask_town)
    if n_t > 0:
        living_sqft[mask_town] = np.clip(np.random.normal(1750, 450, n_t), 900, 4000).astype(int)
        lot_size_sqft[mask_town] = np.clip(np.random.normal(2200, 600, n_t), 1000, 5000).round()
        stories[mask_town] = np.random.choice([2, 3], size=n_t, p=[0.7, 0.3])
        
    # Multi-Family
    n_m = np.sum(mask_multi)
    if n_m > 0:
        living_sqft[mask_multi] = np.clip(np.random.normal(3200, 900, n_m), 1500, 7500).astype(int)
        lot_size_sqft[mask_multi] = np.clip(np.random.normal(6500, 2500, n_m), 2500, 20000).round()
        stories[mask_multi] = np.random.choice([2, 3], size=n_m, p=[0.6, 0.4])
        
    # Single Family
    n_s = np.sum(mask_single)
    if n_s > 0:
        living_sqft[mask_single] = np.clip(np.random.normal(2350, 750, n_s), 750, 6500).astype(int)
        lot_size_sqft[mask_single] = np.clip(np.random.normal(8500, 3500, n_s), 1500, 45000).round()
        stories[mask_single] = np.random.choice([1, 2, 3], size=n_s, p=[0.45, 0.50, 0.05])
        
    # Bedrooms & Bathrooms
    base_beds = np.clip(np.round(living_sqft / 650.0), 1, 6).astype(int)
    bed_noise = np.random.choice([-1, 0, 1], size=n_samples, p=[0.2, 0.6, 0.2])
    bedrooms = np.clip(base_beds + bed_noise, 1, 7).astype(int)
    
    baths_raw = bedrooms * 0.75 + np.random.uniform(-0.5, 0.75, n_samples)
    bathrooms = np.clip(np.round(baths_raw * 2.0) / 2.0, 1.0, 6.0)
    
    rooms_total = bedrooms + bathrooms.astype(int) + np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.1, 0.4, 0.4, 0.1])
    
    # Year Built & Age
    p_years = np.linspace(0.005, 0.02, 80)
    p_years = p_years / np.sum(p_years)
    year_built = np.random.choice(np.arange(1945, 2025), size=n_samples, p=p_years)
    property_age = 2026 - year_built
    
    # Renovation
    renovation_prob = np.where(property_age > 15, np.minimum(0.65, property_age * 0.012), 0.05)
    has_renov_rand = np.random.rand(n_samples) < renovation_prob
    has_been_renovated = np.where(has_renov_rand, "Yes", "No")
    
    rand_renov_year = np.random.randint(2005, 2025, size=n_samples)
    renovation_year = np.where(has_renov_rand, np.maximum(year_built, rand_renov_year), year_built)
    renovation_age = np.where(has_renov_rand, 2026 - renovation_year, property_age)
    
    # Condition
    cond_choices = np.array(PROPERTY_CONDITIONS)
    cond_idx = np.random.choice(len(cond_choices), size=n_samples, p=[0.05, 0.15, 0.50, 0.24, 0.06])
    # Boost condition for newer/renovated homes
    new_mask = (property_age < 10) | ((has_been_renovated == "Yes") & (renovation_age < 7))
    if np.any(new_mask):
        cond_idx[new_mask] = np.random.choice(len(cond_choices), size=np.sum(new_mask), p=[0.01, 0.04, 0.35, 0.45, 0.15])
    property_condition = cond_choices[cond_idx]
    
    # Amenities
    garage_prob = np.where(mask_condo, 0.55, 0.85)
    has_garage_bool = np.random.rand(n_samples) < garage_prob
    has_garage = np.where(has_garage_bool, "Yes", "No")
    garage_spaces = np.where(has_garage_bool, np.random.choice([1, 2, 3], size=n_samples, p=[0.3, 0.55, 0.15]), 0)
    
    warm_cities = np.isin(city_arr, ["Miami", "Phoenix"])
    pool_prob = np.where(warm_cities, 0.38, 0.12)
    has_pool_bool = mask_single & (np.random.rand(n_samples) < pool_prob)
    has_pool = np.where(has_pool_bool, "Yes", "No")
    
    cold_cities = np.isin(city_arr, ["Chicago", "Denver", "New York", "Boston"])
    basement_prob = np.where(cold_cities, 0.75, 0.15)
    has_basement_bool = (mask_single | mask_town) & (np.random.rand(n_samples) < basement_prob)
    has_basement = np.where(has_basement_bool, "Yes", "No")
    basement_sqft = np.where(has_basement_bool, (living_sqft * np.random.uniform(0.3, 0.7, n_samples)).astype(int).astype(float), 0.0)
    
    has_garden = np.where(mask_single & (np.random.rand(n_samples) < 0.70), "Yes", "No")
    has_fireplace = np.where(np.random.rand(n_samples) < 0.42, "Yes", "No")
    has_central_ac = np.where(warm_cities | np.isin(city_arr, ["Atlanta", "Austin"]) | (np.random.rand(n_samples) < 0.85), "Yes", "No")
    has_solar_panels = np.where(np.random.rand(n_samples) < 0.14, "Yes", "No")
    
    roof_type = np.random.choice(ROOF_TYPES, size=n_samples, p=[0.60, 0.15, 0.12, 0.05, 0.08])
    heating_type = np.random.choice(HEATING_TYPES, size=n_samples, p=[0.45, 0.25, 0.18, 0.07, 0.05])
    
    # Neighborhood / Market scores
    incomes = np.random.normal(income_mean_arr, income_std_arr)
    neighborhood_median_income = np.round(np.clip(incomes, 35000.0, 240000.0), -2)
    
    income_norm = (neighborhood_median_income - 35000.0) / (240000.0 - 35000.0)
    school_rating = np.clip(np.round(1.0 + income_norm * 8.0 + np.random.normal(0, 1.1, n_samples)), 1, 10).astype(int)
    
    safety_score = np.round(np.clip(35.0 + income_norm * 50.0 + (dist_miles / 35.0) * 10.0 + np.random.normal(0, 8, n_samples), 15.0, 99.0), 1)
    walk_score = np.round(np.clip(95.0 - (dist_miles * 2.8) + np.random.normal(0, 10, n_samples), 5.0, 99.0), 1)
    transit_score = np.round(np.clip(walk_score * 0.85 + np.random.normal(0, 8, n_samples), 5.0, 98.0), 1)
    park_proximity_miles = np.round(np.clip(np.random.exponential(0.8, n_samples) + 0.1, 0.1, 5.0), 2)
    
    local_property_tax_rate = np.round(np.clip(tax_rate_arr + np.random.normal(0, 0.0015, n_samples), 0.004, 0.035), 4)
    
    hoa_fee_monthly = np.zeros(n_samples)
    if n_c > 0:
        hoa_fee_monthly[mask_condo] = np.round(np.clip(np.random.normal(380, 120, n_c), 80, 1100), 0)
    if n_t > 0:
        hoa_fee_monthly[mask_town] = np.round(np.clip(np.random.normal(160, 60, n_t), 0, 500), 0)
    if (n_s + n_m) > 0:
        hoa_fee_monthly[mask_single | mask_multi] = np.random.choice([0.0, 45.0, 95.0, 150.0], size=n_s + n_m, p=[0.60, 0.20, 0.15, 0.05])
        
    days_on_market = np.clip(np.random.exponential(35, n_samples) + 5, 2, 180).astype(int)
    mortgage_rate_at_listing = np.round(np.random.uniform(5.9, 7.4, n_samples), 2)
    energy_efficiency_rating = np.clip(np.round((year_built - 1940) / 8.5 + np.random.normal(0, 1.2, n_samples)), 1, 10).astype(int)
    
    # TARGET VARIABLE GENERATION (PRICE) - Realistic nonlinear formula
    base_living_val = base_price_sqft_arr * (living_sqft ** 0.96)
    lot_val = np.where(lot_size_sqft > 0, np.log1p(lot_size_sqft) * 4600.0, 0.0)
    
    cond_map = {"Poor": 0.72, "Fair": 0.88, "Good": 1.03, "Excellent": 1.26, "Luxury": 1.68}
    cond_mult = np.array([cond_map[c] for c in property_condition])
    
    type_map = {"Single Family": 1.06, "Condo": 0.94, "Townhouse": 0.98, "Multi-Family": 1.22}
    type_mult = np.array([type_map[t] for t in prop_types])
    
    effective_age = np.where(has_renov_rand, (renovation_age * 0.45 + property_age * 0.55), property_age)
    age_decay = np.maximum(0.68, 1.0 - (effective_age * 0.0042))
    
    amenities_val = (
        (garage_spaces * 16000.0) +
        np.where(has_pool_bool, np.where(warm_cities, 42000.0, 24000.0), 0.0) +
        (basement_sqft * 32.0) +
        np.where(has_fireplace == "Yes", 9000.0, 0.0) +
        np.where(has_central_ac == "Yes", 14000.0, 0.0) +
        np.where(has_solar_panels == "Yes", 19000.0, 0.0) +
        np.where(has_garden == "Yes", 8500.0, 0.0)
    )
    
    school_prem = (school_rating - 5.5) * 0.038 * base_living_val
    safety_prem = (safety_score - 55.0) * 0.0035 * base_living_val
    income_prem = (neighborhood_median_income - income_mean_arr) * 1.35
    
    dense_cities = np.isin(city_arr, ["New York", "San Francisco", "Boston", "Seattle", "Chicago"])
    walk_prem = np.where(dense_cities, (walk_score - 50.0) * 450.0, (walk_score - 50.0) * 150.0)
    distance_penalty = - (dist_miles * 0.009 * base_living_val)
    
    raw_price = (
        (base_living_val + lot_val) * cond_mult * type_mult * age_decay
        + amenities_val
        + school_prem
        + safety_prem
        + income_prem
        + walk_prem
        + distance_penalty
    )
    
    noise = np.random.normal(1.0, 0.075, n_samples)
    final_price = np.maximum(75000.0, raw_price * noise)
    price = np.round(final_price / 500.0) * 500.0
    
    df = pd.DataFrame({
        "city": city_arr,
        "state": state_arr,
        "latitude": latitudes,
        "longitude": longitudes,
        "distance_to_downtown_miles": dist_miles,
        "property_type": prop_types,
        "living_area_sqft": living_sqft,
        "lot_size_sqft": lot_size_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "rooms_total": rooms_total,
        "year_built": year_built,
        "property_age": property_age,
        "renovation_age": renovation_age,
        "has_been_renovated": has_been_renovated,
        "property_condition": property_condition,
        "roof_type": roof_type,
        "heating_type": heating_type,
        "has_garage": has_garage,
        "garage_spaces": garage_spaces,
        "has_pool": has_pool,
        "has_basement": has_basement,
        "basement_sqft": basement_sqft,
        "has_garden": has_garden,
        "has_fireplace": has_fireplace,
        "has_central_ac": has_central_ac,
        "has_solar_panels": has_solar_panels,
        "school_rating": school_rating,
        "safety_score": safety_score,
        "walk_score": walk_score,
        "transit_score": transit_score,
        "park_proximity_miles": park_proximity_miles,
        "neighborhood_median_income": neighborhood_median_income,
        "local_property_tax_rate": local_property_tax_rate,
        "hoa_fee_monthly": hoa_fee_monthly,
        "days_on_market": days_on_market,
        "mortgage_rate_at_listing": mortgage_rate_at_listing,
        "energy_efficiency_rating": energy_efficiency_rating,
        TARGET_COL: price
    })
    
    # Introduce small realistic missing values (~1-2%) to test scikit-learn imputer
    mask_hoa = np.random.rand(n_samples) < 0.02
    df.loc[mask_hoa, "hoa_fee_monthly"] = np.nan
    
    mask_basement = np.random.rand(n_samples) < 0.015
    df.loc[mask_basement, "basement_sqft"] = np.nan
    
    mask_roof = np.random.rand(n_samples) < 0.01
    df.loc[mask_roof, "roof_type"] = np.nan
    
    return df
