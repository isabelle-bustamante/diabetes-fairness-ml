# src/training/preprocessing.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_data(df, feature_cols, label_col="diabetes_label", test_size=0.2, random_state=42):
    """
    Generic train/test split used across all worlds & all models.
    """
    X = df[feature_cols]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    """
    Fits a StandardScaler on training data and transforms both splits.
    Returns scaled arrays and the scaler.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return X_train_s, X_test_s, scaler
