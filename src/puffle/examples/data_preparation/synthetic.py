# ABOUTME: Generates a synthetic dataset for demonstrating group fairness issues.
# ABOUTME: Includes biased target generation based on sensitive attributes to test Puffle regularizers.

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def generate_synthetic_data(
    num_samples=1000, num_features=5, bias_strength=0.8, seed=42
):
    """
    Generate a synthetic dataset with a controlled bias.

    Args:
        num_samples (int): Number of samples to generate.
        num_features (int): Number of features (excluding sensitive attribute).
        bias_strength (float): strength of the bias (0.0 to 1.0).
        seed (int): Random seed.

    Returns:
        pd.DataFrame: Synthetic dataset.

    """
    rng = np.random.default_rng(seed)

    # 1. Generate sensitive attribute Z (binary: 0 or 1)
    # Group 0: 60%, Group 1: 40%
    z = rng.binomial(1, 0.4, num_samples)

    # 2. Generate features X
    # Some features are independent, some might be correlated with Z
    x = rng.standard_normal((num_samples, num_features))

    # 3. Generate target Y with bias
    # If Z=0, higher probability of Y=0
    # If Z=1, higher probability of Y=1
    # Y is generated based on a combination of features and Z
    logits = 0.5 * x[:, 0] + 0.3 * x[:, 1] + bias_strength * (z - 0.5)
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs)

    df = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(num_features)])
    df["sensitive_attribute"] = z
    df["target"] = y

    return df


class SyntheticDataset(Dataset):
    def __init__(self, dataframe):
        self.features = torch.tensor(
            dataframe.drop(columns=["sensitive_attribute", "target"]).values,
            dtype=torch.float32,
        )
        self.sensitive_attributes = torch.tensor(
            dataframe["sensitive_attribute"].values, dtype=torch.long
        )
        self.targets = torch.tensor(dataframe["target"].values, dtype=torch.long)
        self.indices = torch.arange(len(dataframe))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.sensitive_attributes[idx],
            self.targets[idx],
            self.indices[idx],
            idx,
        )


if __name__ == "__main__":
    # Demo generation
    df = generate_synthetic_data(num_samples=100)
    print(df.head())
    print("\nGroup Statistics:")
    print(df.groupby("sensitive_attribute")["target"].mean())
