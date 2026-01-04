import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score
)


def evaluate_model(y_true, y_pred, y_prob, sens_test):
    """
    Computes:
      - Overall accuracy, precision, recall
      - Group TPR (recall)
      - Group FNR
      - Group precision
      - EO gaps
    """

    results = {}


    results["accuracy"] = accuracy_score(y_true, y_pred)
    results["precision"] = precision_score(y_true, y_pred)
    results["recall"] = recall_score(y_true, y_pred)


    gender_metrics = {}
    for g in sens_test["gender"].unique():
        mask = sens_test["gender"] == g

        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        y_prob_g = y_prob[mask]

        tpr = recall_score(y_true_g, y_pred_g, zero_division=0)
        fnr = 1 - tpr
        precision_g = precision_score(y_true_g, y_pred_g, zero_division=0)

        gender_metrics[g] = {
            "TPR": tpr,
            "FNR": fnr,
            "Precision": precision_g
        }

    results["gender_metrics"] = gender_metrics
    results["gender_gap"] = max(m["TPR"] for m in gender_metrics.values()) - \
                            min(m["TPR"] for m in gender_metrics.values())


    race_metrics = {}
    for r in sens_test["race"].unique():
        mask = sens_test["race"] == r

        y_true_r = y_true[mask]
        y_pred_r = y_pred[mask]
        y_prob_r = y_prob[mask]

        tpr = recall_score(y_true_r, y_pred_r, zero_division=0)
        fnr = 1 - tpr
        precision_r = precision_score(y_true_r, y_pred_r, zero_division=0)

        race_metrics[r] = {
            "TPR": tpr,
            "FNR": fnr,
            "Precision": precision_r
        }

    results["race_metrics"] = race_metrics
    results["race_gap"] = max(m["TPR"] for m in race_metrics.values()) - \
                          min(m["TPR"] for m in race_metrics.values())

    return results
