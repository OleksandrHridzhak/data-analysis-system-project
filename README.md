# Real Estate Price Prediction PoC 🏠

Interactive real estate price prediction proof-of-concept built with **Streamlit** and **scikit-learn RandomForestRegressor**.

---

## 🌟 Key Features

1. **Modular Architecture**: Clean separation between data generation, ML pipelines, model management, analytics, and presentation layers (`app.py`, `src/pipeline.py`, `src/model_manager.py`, `src/analytics.py`, `src/ui/`).
2. **Synthetic Data Engine (`generate_data.py`)**: Vectorized generator producing ~20,000 realistic residential properties across 10 major US cities with 39 numerical and categorical features, non-linear dependencies, city geographic/economic profiles, realistic noise, and zero target leakage.
3. **Dedicated Training Interface**:
   - Feature selection for training.
   - Interactive `RandomForestRegressor` hyperparameter tuning (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`).
   - Train/test split control.
   - Model execution only on explicit user trigger (**Train Model** button).
   - Rich metrics: **MAE**, **RMSE**, **R²**, **MAPE**.
   - Interactive **Plotly** visualizations: Actual vs. Predicted scatter plot with $y=x$ line, Residuals analysis, and Feature Importance bar chart.
   - Automatic model persistence via `joblib`.
4. **Prediction & In-Depth Analytics Interface**:
   - **Single Property Inference**: Live market valuation from property specifications.
   - **Numerical Sensitivity Analysis**: Visualizing how varying one feature (e.g., living area, school rating) alters price while holding other attributes constant.
   - **Batch Predictions**: Support for sampling or custom CSV uploads, generating predictions, and one-click **CSV export**.
   - **Categorical Aggregations**: Slicing and aggregating by `city`, `state`, `property_type`, `property_condition` with record counts, medians, means, error metrics, interactive charts, and **CSV export**.
   - **Analytics Feature Exploration**: Filtering by cities, property types, and price ranges without retraining the active model.
5. **Architectural Concept Separation**:
   - Training feature selection $\neq$ Analytics feature selection $\neq$ Prediction inputs.
   - Adjusting analytics controls never triggers model retraining.

---

## 🚀 Quickstart

### 1. Requirements & Installation

Python 3.11+ is required.

```bash
pip install -r requirements.txt
```

### 2. Generate Synthetic Dataset

Generate ~20,000 synthetic records:

```bash
python generate_data.py --samples 20000
```

This saves `properties.csv` to the `data/` folder.

### 3. Run Automated Tests

Execute the automated test suite with pytest:

```bash
python -m pytest -v
```

### 4. Launch Streamlit Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📁 Directory Structure

```text
├── data/
│   └── properties.csv               # Synthetic dataset (20,000 records, 40 columns)
├── models/
│   └── trained_model.joblib         # Serialized RandomForest pipeline bundle
├── src/
│   ├── __init__.py
│   ├── config.py                    # Constants, cities, feature definitions, paths
│   ├── data_generator.py            # Vectorized synthetic real estate data generation
│   ├── pipeline.py                  # scikit-learn Pipeline & ColumnTransformer
│   ├── model_manager.py             # Training, evaluation metrics, feature importance, joblib persistence
│   ├── analytics.py                 # Categorical aggregation & numerical sensitivity analysis
│   └── ui/
│       ├── __init__.py
│       ├── components.py            # Interactive Plotly charts & layout helpers
│       ├── training_view.py         # Streamlit view for model training & evaluation
│       └── prediction_view.py       # Streamlit view for prediction, sensitivity & aggregation
├── tests/
│   ├── test_data_generation.py      # Tests for dataset shape, distributions, and leakage
│   ├── test_pipeline.py             # Tests for scikit-learn preprocessing and imputation
│   └── test_model.py                # Tests for training, persistence, metrics, sensitivity
├── generate_data.py                 # Standalone CLI script for dataset generation
├── app.py                           # Main Streamlit application entry point
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation
```

---

## 🔬 ML Pipeline Specifications

- **Numerical Preprocessing**: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
- **Categorical Preprocessing**: `SimpleImputer(strategy='constant', fill_value='missing')` followed by `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.
- **Estimator**: `scikit-learn RandomForestRegressor` with multi-core support (`n_jobs=-1`).
- **Metrics Tracked**:
  - $MAE = \frac{1}{n} \sum |y - \hat{y}|$
  - $RMSE = \sqrt{\frac{1}{n} \sum (y - \hat{y})^2}$
  - $R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$
  - $MAPE = \frac{100\%}{n} \sum \left|\frac{y - \hat{y}}{y}\right|$

