# src/training/train_lr.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def train_logistic_regression(X_train, y_train, X_test, y_test, max_iter=2000):
    """
    Trains a logistic regression model and returns:
    model, accuracy, AUROC, predictions, probabilities.
    """
    lr = LogisticRegression(max_iter=max_iter)
    lr.fit(X_train, y_train)

    pred = lr.predict(X_test)
    probs = lr.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, probs)

    results = {
        "model": lr,
        "pred": pred,
        "probs": probs,
        "accuracy": acc,
        "AUROC": auc
    }
    return results
