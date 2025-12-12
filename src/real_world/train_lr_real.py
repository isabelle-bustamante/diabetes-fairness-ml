import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def train_lr_real(X_train, y_train):
    """
    Trains a simple Logistic Regression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_lr_real(model, X_test, y_test):
    """
    Returns model predictions + accuracy.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return {"y_pred": y_pred, "accuracy": accuracy}
