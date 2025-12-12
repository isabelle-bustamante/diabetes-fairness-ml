from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(df, feature_cols, label_col="diabetes_label", test_size=0.2):
    # split into train/test
    X = df[feature_cols]
    y = df[label_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    # scale the features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    return X_train_s, X_val_s, y_train, y_val, scaler
