import numpy as np
import pandas as pd


def simulate_realistic_world(n_samples: int = 10000, seed: int | None = None) -> pd.DataFrame:
    """
    Simulate a 'realistic' biased world for diabetes prediction.

    Includes:
      - Structural inequality (race -> SES -> lifestyle -> BMI)
      - True physiology: glucose_true, hba1c_true
      - True diabetes label (diabetes_true)
      - Measurement bias (biased glucose_measured, hba1c_measured)
      - Label bias (diabetes_label != diabetes_true for some groups)

    Bias design:
      - race==2: disadvantaged (lower SES), underdiagnosed (label bias), glucose under-estimation
      - gender==0 (female): HbA1c over-estimation, mild overdiagnosis

    Returns columns:
      race, gender, age, ses, bmi,
      glucose_measured, hba1c_measured,
      diabetes_true, diabetes_label
    """
    if seed is not None:
        np.random.seed(seed)

    # Demographics
    race = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.25, 0.15])  # 2 = disadvantaged minority
    gender = np.random.choice([0, 1], n_samples)  # 0=female, 1=male
    age = np.clip(np.random.normal(50, 12, n_samples), 18, 90)

    # SES related to race (structural inequality)
    ses = (
        0.4 * (race == 0)   # advantaged group
        + 0.0 * (race == 1) # middle group
        - 0.6 * (race == 2) # disadvantaged group
    ) + np.random.normal(0, 0.4, n_samples)

    # Lifestyle score (higher is healthier)
    lifestyle = (
        0.6 * ses
        - 0.02 * (age - 50)
        + np.random.normal(0, 0.5, n_samples)
    )

    # BMI influenced by lifestyle + gender
    bmi = (
        27.0
        - 1.3 * lifestyle
        + 1.0 * (gender == 1)           # males slightly higher BMI
        + np.random.normal(0, 3, n_samples)
    )

    # TRUE glucose physiology
    glucose_true = (
        95.0
        + 1.2 * (bmi - 27.0)
        - 2.5 * lifestyle
        + 0.4 * ((age - 50.0) / 10.0)
        + np.random.normal(0, 8, n_samples)
    )

    # TRUE HbA1c physiology
    hba1c_true = (
        5.4
        + 0.11 * (bmi - 27.0)
        - 0.12 * lifestyle
        + 0.05 * ((age - 50.0) / 10.0)
        + np.random.normal(0, 0.25, n_samples)
    )

    # TRUE diabetes label
    logit = -7.0 + 0.06 * glucose_true + 1.2 * (hba1c_true - 5.4)
    prob = 1.0 / (1.0 + np.exp(-logit))
    diabetes_true = np.random.binomial(1, prob, n_samples)

    # ============================================================
    # MEASUREMENT BIAS
    # ============================================================

    glucose_meas = glucose_true.copy()
    hba1c_meas = hba1c_true.copy()

    # Race 2: glucose under-estimation
    glucose_meas[race == 2] -= 15.0

    # Females: HbA1c over-estimation
    hba1c_meas[gender == 0] += 0.25

    # SES-related noise (low SES = worse sensors)
    glucose_meas += np.random.normal(0, 5 + 3 * (ses < -0.3), n_samples)
    hba1c_meas   += np.random.normal(0, 0.15 + 0.07 * (ses < -0.3), n_samples)

    # ============================================================
    # LABEL BIAS (using diabetes_true as ground truth)
    # ============================================================

    diabetes_label = diabetes_true.copy()

    # Race 2 underdiagnosis: 30% of true positives flipped to 0
    mask_r2 = (race == 2) & (diabetes_true == 1)
    drop = np.random.binomial(1, 0.30, mask_r2.sum())
    diabetes_label[mask_r2] = np.maximum(0, diabetes_label[mask_r2] - drop)

    # Female overdiagnosis: 15% of true negatives flipped to 1
    mask_f = (gender == 0) & (diabetes_true == 0)
    add = np.random.binomial(1, 0.15, mask_f.sum())
    diabetes_label[mask_f] = np.minimum(1, diabetes_label[mask_f] + add)

    df = pd.DataFrame({
        "race": race,
        "gender": gender,
        "age": age,
        "ses": ses,
        "bmi": bmi,
        "glucose_measured": glucose_meas,
        "hba1c_measured": hba1c_meas,
        "diabetes_true": diabetes_true,
        "diabetes_label": diabetes_label
    })

    return df
