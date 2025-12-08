import numpy as np
import pandas as pd


def simulate_measurement_bias_world(n_samples: int = 10000, seed: int | None = None) -> pd.DataFrame:
    """
    Simulate a world with fair underlying physiology but biased measurements.

    - diabetes_label is still the true disease status
    - race and gender affect how glucose/hba1c are measured (biased sensors)

    Biases:
        - race == 2: glucose is systematically under-estimated
        - gender == 0 (female): HbA1c is systematically over-estimated
        - race == 1: extra measurement noise

    Columns:
        race, gender, age
        glucose_measured, hba1c_measured
        diabetes_label
    """
    if seed is not None:
        np.random.seed(seed)

    # Demographics
    race = np.random.choice([0, 1, 2], size=n_samples, p=[0.6, 0.25, 0.15])
    gender = np.random.choice([0, 1], size=n_samples)
    age = np.random.normal(50, 10, n_samples).clip(18, 90)

    # True diabetes
    diabetes = np.random.binomial(1, 0.20, n_samples)

    # True underlying labs
    glucose_non = np.random.normal(95, 8, n_samples)
    glucose_diab = np.random.normal(160, 20, n_samples)

    hba1c_non = np.random.normal(5.3, 0.25, n_samples)
    hba1c_diab = np.random.normal(7.0, 0.6, n_samples)

    glucose_true = np.where(diabetes == 1, glucose_diab, glucose_non)
    hba1c_true   = np.where(diabetes == 1, hba1c_diab, hba1c_non)

    # Start measured values as true ones
    glucose_meas = glucose_true.copy()
    hba1c_meas   = hba1c_true.copy()

    # Measurement bias
    # Race 2: glucose underestimation
    glucose_meas[race == 2] -= 15.0

    # Females: HbA1c overestimation
    hba1c_meas[gender == 0] += 0.3

    # Race 1: extra measurement noise
    mask_r1 = (race == 1)
    glucose_meas[mask_r1] += np.random.normal(0, 10, mask_r1.sum())
    hba1c_meas[mask_r1]   += np.random.normal(0, 0.4, mask_r1.sum())

    # Base measurement noise for everyone
    glucose_meas += np.random.normal(0, 5, n_samples)
    hba1c_meas   += np.random.normal(0, 0.15, n_samples)

    df = pd.DataFrame({
        "race": race,
        "gender": gender,
        "age": age,
        "glucose_measured": glucose_meas,
        "hba1c_measured": hba1c_meas,
        "diabetes_label": diabetes
    })

    return df
