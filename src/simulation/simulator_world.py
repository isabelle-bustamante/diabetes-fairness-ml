import numpy as np
import pandas as pd

def generate_diabetes_data(n_samples=5000, seed=42):
    gender = np.random.choice([0,1], size=n_samples, p=[0.5,0.5])
    age = np.random.normal(50, 12, size=n_samples)
    bmi = np.random.normal(30, 5, size=n_samples)
    glucose = np.random.normal(120, 20, size=n_samples)

    # Coefficients small, intercept adjusted to get ~30-50% positives
    logits = 0.05*age + 0.03*bmi + 0.03*glucose + 0.5*gender - 8
    logits += np.random.normal(0, 3, size=n_samples)  # noise

    prob = 1 / (1 + np.exp(-logits))
    diabetes = np.random.binomial(1, prob)

    df = pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'glucose': glucose,
        'gender': gender,
        'diabetes': diabetes
    })
    return df