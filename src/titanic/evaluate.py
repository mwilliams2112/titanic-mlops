from sklearn.metrics import accuracy_score, recall_score, roc_auc_score


def evaluate(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    recalls = recall_score(y_test, y_pred, average=None)  # one value per class

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "recall_class_0": float(recalls[0]),
        "recall_class_1": float(recalls[1]),
    }