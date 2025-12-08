import numpy as np
import pandas as pd

from src.data_generation.fair_world import simulate_fair_world




def simulate_representation_bias_world(
    n_samples: int = 10000,
    minority_race: int = 2,
    minority_frac: float = 0.1,
    seed: int | None = None
) -> pd.DataFrame:
    """
    Simulate a world with representation bias (under-sampled minority group).

    Steps:
      1. Generate a fair-world dataset.
      2. Under-sample the chosen minority race to `minority_frac` of its fair proportion.
    """

    # Create the full dataset
    df = simulate_fair_world(n_samples=n_samples, seed=seed)

    # Split into minority group and majority group
    minority_group = df[df["race"] == minority_race]
    majority_group = df[df["race"] != minority_race]

    # Decide how many minority samples to keep
    num_minority_to_keep = int(len(minority_group) * minority_frac)

    if num_minority_to_keep < 1:
        num_minority_to_keep = 1

    # Random minority samples
    minority_sampled = minority_group.sample(
        n=num_minority_to_keep,
        random_state=seed
    )

    # Combine majority with under-sampled minority group
    combined = pd.concat([majority_group, minority_sampled])

    # Shuffle rows
    combined = combined.sample(frac=1.0, random_state=seed)

    # Reset index numbers
    combined = combined.reset_index(drop=True)

    return combined
