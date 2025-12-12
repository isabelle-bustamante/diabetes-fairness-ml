import numpy as np
import pandas as pd

def simulate_realistic_world_simple(n_samples=10000, seed=None):

    if seed is not None:
        np.random.seed(seed)

    # Demographics
    race = np.random.choice([0,1,2], n_samples, p=[0.6,0.25,0.15])
    gender = np.random.choice([0,1], n_samples)  # 0=female, 1=male
    age = np.random.normal(50, 12, n_samples)

    #  SES depends on race
    ses = (
        0.5*(race==0)
        + 0.0*(race==1)
        - 0.5*(race==2)
        + np.random.normal(0, 0.3, n_samples)
    )

    # BMI depends on SES and gender
    bmi = (
        27
        + 1.0*(gender==1)
        - 1.2*ses
        + np.random.normal(0, 2, n_samples)
    )

    # Labs depend on BMI
    glucose = 95 + 2*(bmi-27) + np.random.normal(0, 6, n_samples)
    hba1c   = 5.4 + 0.1*(bmi-27) + np.random.normal(0, 0.2, n_samples)

    # Disease risk from labs
    risk = (
        0.03*(glucose - 110)
        + 0.5*(hba1c - 5.8)
        + 0.2*(bmi - 27)
        + np.random.normal(0, 0.5, n_samples)
    )

    diabetes_prob = 1/(1 + np.exp(-risk))
    diabetes_true = np.random.binomial(1, diabetes_prob)

    #  for label bias
    diabetes_label = diabetes_true.copy()

    # Race 2 underdiagnosed
    mask_r2 = (race == 2) & (diabetes_true == 1)
    diabetes_label[mask_r2] -= np.random.binomial(1, 0.25, mask_r2.sum())

    # Females overdiagnosed
    mask_f = (gender == 0) & (diabetes_true == 0)
    diabetes_label[mask_f] += np.random.binomial(1, 0.15, mask_f.sum())

    df = pd.DataFrame({
        "race": race,
        "gender": gender,
        "age": age,
        "ses": ses,
        "bmi": bmi,
        "glucose": glucose,
        "hba1c": hba1c,
        "diabetes_true": diabetes_true,
        "diabetes_label": diabetes_label
    })

    return df
