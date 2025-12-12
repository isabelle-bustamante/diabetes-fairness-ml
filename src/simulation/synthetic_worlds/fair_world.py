import numpy as np
import pandas as pd


def simulate_fair_world(n_samples: int = 10000, seed: int | None = None) -> pd.DataFrame:
    """
    Simulate a 'fair' world for a diabetes-like prediction task.

    - race, gender, age are generated
    - diabetes_label is the true underlying diabetes status
    - lab measurements (glucose_measured, hba1c_measured) are unbiased

    Columns:
        race:   {0,1,2}
        gender: {0=female, 1=male}
        age:    float
        glucose_measured: float
        hba1c_measured:   float
        diabetes_label:   {0,1}
    """
    if seed is not None:
        np.random.seed(seed)

    # Demographics
    race = np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.25, 0.15])
    gender = np.random.choice([0, 1], size=n_samples)
    age = np.random.normal(50, 10, n_samples).clip(18, 90)

    # Prevalence ~20%
    diabetes = np.random.binomial(1, 0.20, size=n_samples)

    # Non-diabetic labs
    glucose_non = np.random.normal(95, 8, n_samples)
    hba1c_non   = np.random.normal(5.3, 0.25, n_samples)

    # Diabetic labs
    glucose_diab = np.random.normal(160, 20, n_samples)
    hba1c_diab   = np.random.normal(7.0, 0.6, n_samples)

    glucose = np.where(diabetes == 1, glucose_diab, glucose_non)
    hba1c   = np.where(diabetes == 1, hba1c_diab, hba1c_non)

    # Unbiased measurement noise
    glucose_meas = glucose + np.random.normal(0, 5, n_samples)
    hba1c_meas   = hba1c   + np.random.normal(0, 0.15, n_samples)

    df = pd.DataFrame({
        "race": race,
        "gender": gender,
        "age": age,
        "glucose_measured": glucose_meas,
        "hba1c_measured": hba1c_meas,
        "diabetes_label": diabetes
    })

    return df
