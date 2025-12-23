# src/simulation/train_lr_sim.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_logistic_regression(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # P(Y=1)

    acc = accuracy_score(y_test, y_pred)

    return {
        "model": model,
        "accuracy": acc,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

