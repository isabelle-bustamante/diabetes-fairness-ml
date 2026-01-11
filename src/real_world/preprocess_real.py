import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def prepare_real_dataset(df):
    """
       Prepares the real-world diabetes dataset for modeling by:
       - Cleaning and filtering records
       - Separating sensitive attributes for fairness evaluation
       - Applying a unified preprocessing pipeline
       - Returning stratified train/test splits
       """


    race_cols = [
        "race:AfricanAmerican",
        "race:Asian",
        "race:Other",
        "race:Hispanic",
        "race:Caucasian"
    ]

    # Assign the race with highest indicator
    df["race"] = df[race_cols].idxmax(axis=1).str.replace("race:", "")
    df = df.drop(columns=race_cols)

    #  Remove columns we cannot one-hot encode
    # We remove clinical_notes because free text would explode feature count
    df = df.drop(columns=["year", "location", "clinical_notes"])


    # Keep only Male/Female
    df = df[df["gender"].isin(["Male", "Female"])]

    # Split features, target, sensitive columns
    TARGET_COL = "diabetes"
    SENSITIVE_COLS = ["gender", "race"]

    sensitive = df[SENSITIVE_COLS].copy()
    X = df.drop(columns=[TARGET_COL] + SENSITIVE_COLS)
    y = df[TARGET_COL]

    #  Train/test split with stratification
    X_temp, X_test, y_temp, y_test, sens_temp, sens_test = train_test_split(
        X, y, sensitive,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    X_train, X_val, y_train, y_val, sens_train, sens_val = train_test_split(
        X_temp, y_temp, sens_temp,
        test_size=0.2,  # 20% of remaining 70% → 14% total
        random_state=42,
        stratify=y_temp
    )

    #  Define numeric and categorical columns
    numeric_features = ["age", "bmi", "hbA1c_level", "blood_glucose_level"]

    categorical_features = X.columns.difference(numeric_features)

    #  Preprocessing: scale numerics, one-hot encode categoricals
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    #  Transform datasets
    X_train_processed = preprocess.fit_transform(X_train)
    X_val_processed = preprocess.transform(X_val)
    X_test_processed = preprocess.transform(X_test)

    return {
        "X_train": X_train_processed,
        "X_val": X_val_processed,
        "X_test": X_test_processed,
        "y_train": y_train.reset_index(drop=True),
        "y_val": y_val.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "sens_train": sens_train.reset_index(drop=True),
        "sens_val": sens_val.reset_index(drop=True),
        "sens_test": sens_test.reset_index(drop=True),
        "preprocess": preprocess,
        "feature_names": X.columns.tolist(),
    }
