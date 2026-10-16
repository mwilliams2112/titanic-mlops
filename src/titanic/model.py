from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        [("sex", OneHotEncoder(handle_unknown="ignore"), ["Sex"])],
        remainder="passthrough",
    )
    return Pipeline([
        ("preprocess", preprocess),
        ("model", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)),
    ])