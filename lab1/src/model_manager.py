import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.pipeline import build_pipeline
from src.config import MODEL_FILE, TARGET_COL


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes regression evaluation metrics: MAE, RMSE, R2, MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Avoid zero-division in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    
    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": float(mape)
    }


def get_feature_names_out(pipeline) -> List[str]:
    """
    Extracts transformed feature names from ColumnTransformer.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = []
    
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder" or trans == "drop":
            continue
        if hasattr(trans, "get_feature_names_out"):
            sub_names = trans.get_feature_names_out(cols)
            feature_names.extend(sub_names)
        elif hasattr(trans, "named_steps") and "onehot" in trans.named_steps:
            sub_names = trans.named_steps["onehot"].get_feature_names_out(cols)
            feature_names.extend(sub_names)
        else:
            feature_names.extend(cols)
            
    return feature_names


def extract_feature_importances(pipeline, top_n: int = 25) -> pd.DataFrame:
    """
    Returns sorted feature importances from trained RandomForestRegressor.
    """
    regressor = pipeline.named_steps["regressor"]
    importances = regressor.feature_importances_
    
    feature_names = get_feature_names_out(pipeline)
    if len(feature_names) != len(importances):
        feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    return df_imp.head(top_n)


def train_and_evaluate(
    df: pd.DataFrame,
    numerical_features: List[str],
    categorical_features: List[str],
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: Optional[int] = 16,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    max_features: str = "sqrt",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains the ML pipeline and evaluates performance on train & test sets.
    """
    all_features = numerical_features + categorical_features
    if not all_features:
        raise ValueError("At least one feature must be selected for training.")
        
    X = df[all_features].copy()
    y = df[TARGET_COL].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    pipeline = build_pipeline(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state
    )
    
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)
    
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    # Feature importances
    df_importance = extract_feature_importances(pipeline, top_n=20)
    
    # Evaluation dataframe with actual vs predicted and residuals
    df_eval = X_test.copy()
    df_eval["Actual_Price"] = y_test
    df_eval["Predicted_Price"] = y_pred_test
    df_eval["Residual"] = y_test - y_pred_test
    df_eval["Abs_Error"] = np.abs(df_eval["Residual"])
    df_eval["Pct_Error"] = (df_eval["Abs_Error"] / df_eval["Actual_Price"]) * 100.0
    
    bundle = {
        "pipeline": pipeline,
        "numerical_features": numerical_features,
        "categorical_features": categorical_features,
        "all_features": all_features,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "df_importance": df_importance,
        "eval_sample": df_eval,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "test_size": test_size,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
    }
    
    return bundle


def save_model_bundle(bundle: Dict[str, Any], filepath: Path = MODEL_FILE) -> str:
    """
    Serializes the model bundle to disk using joblib.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, filepath)
    return str(filepath)


def load_model_bundle(filepath: Path = MODEL_FILE) -> Optional[Dict[str, Any]]:
    """
    Loads serialized model bundle from disk using joblib.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    return joblib.load(filepath)

