# src/metrics/fairness.py

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# ----------------------------------------------------
# 1. GROUP FAIRNESS METRICS (FPR, FNR, TPR, TNR)
# ----------------------------------------------------
def fairness_by_group(y_true, y_pred, group_values):
    """
    Computes FPR, FNR, TPR, TNR for each subgroup.
    """
    results = {}

    for group in sorted(np.unique(group_values)):
        mask = (group_values == group)

        tn, fp, fn, tp = confusion_matrix(
            y_true[mask], y_pred[mask], labels=[0, 1]
        ).ravel()

        results[group] = {
            "FPR": fp / (fp + tn) if (fp + tn) > 0 else np.nan,
            "FNR": fn / (fn + tp) if (fn + tp) > 0 else np.nan,
            "TPR": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
            "TNR": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
        }

    return pd.DataFrame(results).T


# ----------------------------------------------------
# 2. EQUAL OPPORTUNITY (EO)
#    EO gap = |TPR_group1 - TPR_group2|
# ----------------------------------------------------
def equal_opportunity_diff(fairness_df):
    """
    Computes Equal Opportunity difference:
        EO = max(TPR) - min(TPR)
    """
    tprs = fairness_df["TPR"].dropna()
    return float(tprs.max() - tprs.min())


# ----------------------------------------------------
# 3. EQUALIZED ODDS (EOdds)
#    EO = |TPR_diff| + |FPR_diff|
# ----------------------------------------------------
def equalized_odds_diff(fairness_df):
    """
    Computes Equalized Odds difference:
        EO = (max(TPR) - min(TPR)) + (max(FPR) - min(FPR))
    """
    tprs = fairness_df["TPR"].dropna()
    fprs = fairness_df["FPR"].dropna()

    tpr_gap = float(tprs.max() - tprs.min())
    fpr_gap = float(fprs.max() - fprs.min())

    return tpr_gap + fpr_gap
