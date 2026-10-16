import joblib
import mlflow
import mlflow.sklearn
from sklearn.dummy import DummyClassifier

from titanic.config import MODEL_PARAMS, MODEL_PATH
from titanic.data import download_data, split_data
from titanic.evaluate import evaluate
from titanic.model import build_pipeline

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("titanic")


def main():
    data = download_data()
    X_train, X_test, y_train, y_test = split_data(data)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    baseline_metrics = evaluate(baseline, X_test, y_test)
    model_metrics = evaluate(pipeline, X_test, y_test)

    with mlflow.start_run():
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_metrics(model_metrics)
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_metrics.items()})
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name="titanic-survival",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

    print("baseline:", baseline_metrics)
    print("model:   ", model_metrics)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()