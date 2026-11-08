from typing import List, Optional
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor


def build_pipeline(
    numerical_features: List[str],
    categorical_features: List[str],
    n_estimators: int = 100,
    max_depth: Optional[int] = 16,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    max_features: str = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1
) -> Pipeline:
    """
    Constructs a scikit-learn Pipeline with ColumnTransformer,
    handling missing values, one-hot encoding, scaling, and RandomForestRegressor.
    """
    transformers = []
    
    if numerical_features:
        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", num_transformer, numerical_features))
        
    if categorical_features:
        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])
        transformers.append(("cat", cat_transformer, categorical_features))
        
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )
    
    regressor = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])
    
    return pipeline

