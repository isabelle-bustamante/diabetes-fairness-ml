import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def prepare_real_dataset(df):
    """
    Cleans the dataset, creates race/gender variables,
    removes text fields, performs preprocessing,
    and returns train/test splits with consistent features.
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
    df = df.drop(columns=["location", "clinical_notes"])

    # Keep only Male/Female
    df = df[df["gender"].isin(["Male", "Female"])]

    # Split features, target, sensitive columns
    TARGET_COL = "diabetes"
    SENSITIVE_COLS = ["gender", "race"]

    sensitive = df[SENSITIVE_COLS].copy()
    X = df.drop(columns=[TARGET_COL] + SENSITIVE_COLS)
    y = df[TARGET_COL]

    #  Train/test split with stratification
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X, y, sensitive,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    #  Define numeric and categorical columns
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object", "category", "bool"]).columns

    #  Preprocessing: scale numerics, one-hot encode categoricals
    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
        ]
    )

    #  Transform datasets
    X_train_processed = preprocess.fit_transform(X_train)
    X_test_processed  = preprocess.transform(X_test)

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train.reset_index(drop=True),
        "y_test": y_test.reset_index(drop=True),
        "sens_train": sens_train.reset_index(drop=True),
        "sens_test": sens_test.reset_index(drop=True),
        "preprocess": preprocess,
        "feature_names": X.columns.tolist(),
    }
