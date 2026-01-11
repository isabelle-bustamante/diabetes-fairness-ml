import numpy as np
from sklearn.linear_model import LogisticRegression


def train_lr_real(X_train, y_train):
    """
    Trains a logistic regression model to learn diabetes risk.
    Outputs probabilities.
    """
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )
    model.fit(X_train, y_train)
    return model


def evaluate_lr_real(model, X_test):
    """
    Returns predicted probabilities of diabetes for each test sample.
    """
    y_pred_probs = model.predict_proba(X_test)[:, 1]
    return {
        "y_pred_probs": y_pred_probs
    }
