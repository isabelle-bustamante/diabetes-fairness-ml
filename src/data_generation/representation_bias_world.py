import numpy as np
import pandas as pd

from .fair_world import simulate_fair_world


def simulate_representation_bias_world(
    n_samples: int = 10000,
    minority_race: int = 2,
    minority_frac: float = 0.1,
    seed: int | None = None
) -> pd.DataFrame:
    """
    Simulate a world with REPRESENTATION BIAS (under-sampled minority group),
    but otherwise fair physiology and measurements.

    Steps:
      1. Generate a fair-world dataset.
      2. Under-sample the chosen minority race to `minority_frac` of its fair proportion.

    Args:
        n_samples: total number of individuals to simulate before sub-sampling
        minority_race: which race code to under-sample (default=2)
        minority_frac: fraction of that group's fair-world size to keep (0<frac<=1)
        seed: optional random seed

    Returns:
        DataFrame with same columns as fair world, but imbalanced race distribution.
    """
    if seed is not None:
        np.random.seed(seed)

    df_full = simulate_fair_world(n_samples=n_samples, seed=seed)

    mask_min = (df_full["race"] == minority_race)
    df_min = df_full[mask_min]
    df_maj = df_full[~mask_min]

    # Under-sample minority group
    keep_n = max(1, int(len(df_min) * minority_frac))
    df_min_under = df_min.sample(n=keep_n, random_state=seed)

    df_underrep = pd.concat([df_maj, df_min_under]).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)

    return df_underrep
