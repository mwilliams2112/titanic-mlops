import joblib
from sklearn.dummy import DummyClassifier

from titanic.config import MODEL_PATH
from titanic.data import download_data, split_data
from titanic.evaluate import evaluate
from titanic.model import build_pipeline


def main():
    data = download_data()
    X_train, X_test, y_train, y_test = split_data(data)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    print("baseline:", evaluate(baseline, X_test, y_test))
    print("model:   ", evaluate(pipeline, X_test, y_test))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()