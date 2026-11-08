from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


def compute_categorical_aggregation(
    df: pd.DataFrame,
    group_col: str,
    actual_col: str = "Actual_Price",
    pred_col: str = "Predicted_Price"
) -> pd.DataFrame:
    """
    Computes summary aggregations for a categorical feature:
    Record count, Mean/Median Actual, Mean/Median Predicted, and Prediction Errors.
    """
    if group_col not in df.columns:
        raise ValueError(f"Column '{group_col}' not found in dataframe.")
        
    # Grouping
    grouped = df.groupby(group_col)
    
    records = []
    for name, group in grouped:
        count = len(group)
        mean_act = group[actual_col].mean()
        med_act = group[actual_col].median()
        mean_pred = group[pred_col].mean()
        med_pred = group[pred_col].median()
        mae = np.mean(np.abs(group[actual_col] - group[pred_col]))
        mape = np.mean(np.abs(group[actual_col] - group[pred_col]) / group[actual_col]) * 100.0
        
        records.append({
            group_col: name,
            "Count": count,
            "Mean Actual ($)": round(mean_act, 0),
            "Median Actual ($)": round(med_act, 0),
            "Mean Predicted ($)": round(mean_pred, 0),
            "Median Predicted ($)": round(med_pred, 0),
            "MAE ($)": round(mae, 0),
            "MAPE (%)": round(mape, 2)
        })
        
    res_df = pd.DataFrame(records)
    return res_df.sort_values(by="Count", ascending=False).reset_index(drop=True)


def compute_numerical_sensitivity(
    pipeline,
    base_property: Dict[str, Any],
    feature_name: str,
    min_val: float,
    max_val: float,
    steps: int = 40
) -> pd.DataFrame:
    """
    Generates sensitivity prediction curve for a chosen numerical feature
    while keeping all other base property features constant.
    """
    vals = np.linspace(min_val, max_val, steps)
    
    # Duplicate base property dictionary across steps
    batch_records = []
    for v in vals:
        row = dict(base_property)
        row[feature_name] = v
        batch_records.append(row)
        
    batch_df = pd.DataFrame(batch_records)
    predictions = pipeline.predict(batch_df)
    
    return pd.DataFrame({
        feature_name: vals,
        "Predicted_Price": np.round(predictions, 0)
    })

