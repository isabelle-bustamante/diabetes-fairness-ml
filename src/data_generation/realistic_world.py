import numpy as np
import pandas as pd


def simulate_realistic_world(n_samples: int = 10000, seed: int | None = None) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)

    # ---------------------------------------------------
    # 1. Demographics
    # ---------------------------------------------------
    race = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.25, 0.15])
    gender = np.random.choice([0, 1], n_samples)  # 0 = female, 1 = male
    age = np.clip(np.random.normal(50, 12, n_samples), 18, 90)

    # ---------------------------------------------------
    # 2. SES (lower for race 2)
    # ---------------------------------------------------
    ses = (
        0.5 * (race == 0)
        + 0.0 * (race == 1)
        - 0.5 * (race == 2)
        + np.random.normal(0, 0.3, n_samples)
    )

    # ---------------------------------------------------
    # 3. BMI (SES + gender + a bit nonlinear)
    # ---------------------------------------------------
    bmi = (
        27
        + 1.2 * (gender == 1)      # males slightly higher BMI
        - 1.0 * ses                # low SES → higher BMI
        + 0.15 * ses**2            # mild curve
        + np.random.normal(0, 2, n_samples)
    )

    # ---------------------------------------------------
    # 4. Glucose & HbA1c physiology (nonlinear but learnable)
    # ---------------------------------------------------
    glucose_true = (
        95
        + 1.8 * (bmi - 27)
        + 0.10 * (bmi - 27)**2     # clear nonlinearity
        + np.random.normal(0, 7, n_samples)
    )

    hba1c_true = (
        5.4
        + 0.10 * (bmi - 27)
        + np.minimum(0.3, 0.015 * (bmi - 27))   # mild curved increase
        + np.random.normal(0, 0.22, n_samples)
    )

    # ---------------------------------------------------
    # 5. STRONGER Diabetes mechanism = NN advantage appears!
    # ---------------------------------------------------
    risk = (
            0.02 * (glucose_true - 100) ** 2  # strong nonlinearity
            + 1.2 * (hba1c_true - 5.8) ** 3  # cubic effect
            + 0.6 * (bmi - 27) * (hba1c_true - 5.6)  # interaction
            + 0.4 * ses * (glucose_true - 110)  # SES × glucose effect
            + np.random.normal(0, 0.3, n_samples)  # label noise
    )

    # convert risk → probability (bounded between 0 and 1)
    diabetes_prob = 1 / (1 + np.exp(-risk))

    # sample the true label
    diabetes_true = np.random.binomial(1, diabetes_prob)

    # ---------------------------------------------------
    # 6. Measurement bias (fairness issue!)
    # ---------------------------------------------------
    glucose_measured = glucose_true.copy()
    hba1c_measured = hba1c_true.copy()

    # Race 2 always gets glucose underestimated → unfair
    glucose_measured[race == 2] -= 8

    # Females get HbA1c overestimated → unfair
    hba1c_measured[gender == 0] += 0.15

    # SES noise → poorer groups get worse measurements
    low_ses = ses < -0.3
    glucose_measured += np.random.normal(0, 5 + 3 * low_ses, n_samples)
    hba1c_measured += np.random.normal(0, 0.12 + 0.05 * low_ses, n_samples)

    # ---------------------------------------------------
    # 7. LABEL BIAS (small amount — realistic)
    # ---------------------------------------------------
    diabetes_label = diabetes_true.copy()

    # Race 2 underdiagnosed
    r2_mask = (race == 2) & (diabetes_true == 1)
    diabetes_label[r2_mask] = np.maximum(
        0,
        diabetes_label[r2_mask] - np.random.binomial(1, 0.25, r2_mask.sum())
    )

    # Females overdiagnosed
    f_mask = (gender == 0) & (diabetes_true == 0)
    diabetes_label[f_mask] = np.minimum(
        1,
        diabetes_label[f_mask] + np.random.binomial(1, 0.15, f_mask.sum())
    )

    # ---------------------------------------------------
    # 8. Final DataFrame
    # ---------------------------------------------------
    df = pd.DataFrame({
        "race": race,
        "gender": gender,
        "age": age,
        "ses": ses,
        "bmi": bmi,
        "glucose_true": glucose_true,
        "hba1c_true": hba1c_true,
        "glucose_measured": glucose_measured,
        "hba1c_measured": hba1c_measured,
        "diabetes_true": diabetes_true,
        "diabetes_label": diabetes_label
    })

    return df
