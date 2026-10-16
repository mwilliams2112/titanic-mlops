from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from titanic.config import CATEGORICAL_FEATURES, MODEL_PARAMS, NUMERIC_FEATURES


def build_pipeline(params: dict | None = None) -> Pipeline:
    params = params or MODEL_PARAMS
    preprocess = ColumnTransformer(
        [
            ("sex", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(**params)),
    ])