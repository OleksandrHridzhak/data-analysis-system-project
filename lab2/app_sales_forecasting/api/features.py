"""
Feature engineering and preprocessing pipeline for sales forecasting.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


class FeaturePreprocessor:
    """Encapsulates scikit-learn ColumnTransformer fitted strictly on training data."""

    def __init__(self, selected_features: List[str]):
        self.selected_features = list(selected_features)
        
        all_num = ["Promo", "SchoolHoliday", "logMonthSales", "CompetitionDistance"]
        all_cat = ["Store", "StateHoliday", "StoreType", "Assortment", "weekday", "monthday", "month"]
        
        self.num_cols = [c for c in self.selected_features if c in all_num]
        self.cat_cols = [c for c in self.selected_features if c in all_cat]
        
        transformers = []
        if self.num_cols:
            num_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median"))
            ])
            transformers.append(("num", num_pipe, self.num_cols))
            
        if self.cat_cols:
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])
            transformers.append(("cat", cat_pipe, self.cat_cols))
            
        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )
        self.is_fitted = False
        self.transformed_feature_names: List[str] = []

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        """Fits transformers on training features."""
        # Ensure correct column dtypes
        X_copy = X.copy()
        for c in self.cat_cols:
            if c in X_copy.columns:
                X_copy[c] = X_copy[c].astype(str)
                
        self.column_transformer.fit(X_copy)
        self.is_fitted = True
        
        # Build feature names
        names = []
        if self.num_cols:
            names.extend(self.num_cols)
        if self.cat_cols:
            ohe = self.column_transformer.named_transformers_["cat"].named_steps["ohe"]
            cat_names = ohe.get_feature_names_out(self.cat_cols).tolist()
            names.extend(cat_names)
        self.transformed_feature_names = names
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transforms features using fitted pipeline."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePreprocessor is not fitted yet.")
        X_copy = X.copy()
        for c in self.cat_cols:
            if c in X_copy.columns:
                X_copy[c] = X_copy[c].astype(str)
        return self.column_transformer.transform(X_copy)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.fit(X).transform(X)

    def aggregate_feature_importances(self, raw_importances: np.ndarray) -> Dict[str, float]:
        """
        Aggregates one-hot encoded feature importances back to original features.
        """
        if len(raw_importances) != len(self.transformed_feature_names):
            return {f: float(1.0 / len(self.selected_features)) for f in self.selected_features}

        importance_by_orig = {f: 0.0 for f in self.selected_features}
        for name, imp in zip(self.transformed_feature_names, raw_importances):
            # check which original feature this corresponds to
            matched = False
            for orig in self.selected_features:
                if name == orig or name.startswith(f"{orig}_"):
                    importance_by_orig[orig] += float(imp)
                    matched = True
                    break
            if not matched:
                # fallback
                importance_by_orig[self.selected_features[0]] += float(imp)

        # Normalize to sum to 1.0
        total = sum(importance_by_orig.values())
        if total > 0:
            for k in importance_by_orig:
                importance_by_orig[k] = float(np.round(importance_by_orig[k] / total, 4))

        # Sort descending
        return dict(sorted(importance_by_orig.items(), key=lambda item: item[1], reverse=True))

