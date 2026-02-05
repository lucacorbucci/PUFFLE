# ruff: noqa: S101, T201, PERF401, PLR2004, RUF043
"""Test binary validation in FairnessPartitioner."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

sys.path.append(str(Path(__file__).parent.parent / "src"))

from FlowerFLTemplate.Datasets.Partitioner.fairness_partitioner import (
    FairnessPartitioner,
)


def test_binary_validation_passes_with_binary_data():
    """Test that validation passes when attributes are binary."""
    # Create binary dataset
    data = []
    for _ in range(100):
        data.append({"target": 0, "sensitive": 0})
    for _ in range(100):
        data.append({"target": 0, "sensitive": 1})
    for _ in range(100):
        data.append({"target": 1, "sensitive": 0})
    for _ in range(100):
        data.append({"target": 1, "sensitive": 1})

    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)

    # Should not raise an error
    partitioner = FairnessPartitioner(
        num_partitions=10,
        sensitive_attribute="sensitive",
        target_attribute="target",
        ratio_unfair_clients=0.5,
        group_to_reduce=(0, 0),
        ratio_unfairness=(0.1, 0.1),
        dataset=dataset,
        seed=42,
    )

    assert partitioner is not None
    print("SUCCESS: Binary validation passed with binary data")


def test_binary_validation_fails_with_non_binary_sensitive():
    """Test that validation fails when sensitive attribute is not binary."""
    # Create dataset with non-binary sensitive attribute
    data = []
    for _ in range(100):
        data.append({"target": 0, "sensitive": 0})
    for _ in range(100):
        data.append({"target": 0, "sensitive": 1})
    for _ in range(100):
        data.append({"target": 1, "sensitive": 2})  # Non-binary value

    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)

    # Should raise ValueError
    with pytest.raises(
        ValueError, match="sensitive_attribute.*must be binary.*Found values.*0.*1.*2"
    ):
        FairnessPartitioner(
            num_partitions=10,
            sensitive_attribute="sensitive",
            target_attribute="target",
            ratio_unfair_clients=0.5,
            group_to_reduce=(0, 0),
            ratio_unfairness=(0.1, 0.1),
            dataset=dataset,
            seed=42,
        )

    print(
        "SUCCESS: Binary validation correctly rejected non-binary sensitive attribute"
    )


def test_binary_validation_fails_with_non_binary_target():
    """Test that validation fails when target attribute is not binary."""
    # Create dataset with non-binary target attribute
    data = []
    for _ in range(100):
        data.append({"target": 0, "sensitive": 0})
    for _ in range(100):
        data.append({"target": 1, "sensitive": 1})
    for _ in range(100):
        data.append({"target": 5, "sensitive": 0})  # Non-binary value

    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)

    # Should raise ValueError
    with pytest.raises(
        ValueError, match="target_attribute.*must be binary.*Found values.*0.*1.*5"
    ):
        FairnessPartitioner(
            num_partitions=10,
            sensitive_attribute="sensitive",
            target_attribute="target",
            ratio_unfair_clients=0.5,
            group_to_reduce=(0, 0),
            ratio_unfairness=(0.1, 0.1),
            dataset=dataset,
            seed=42,
        )

    print("SUCCESS: Binary validation correctly rejected non-binary target attribute")


def test_dutch_preprocessing_creates_binary_attributes():
    """Test that Dutch dataset preprocessing creates binary attributes."""
    # Simulate Dutch dataset
    data = {
        "sex": [1, 2, 1, 2, 1],  # 1=male, 2=female
        "occupation": [100, 600, 300, 700, 450],  # threshold at 500
    }
    df = pd.DataFrame(data)

    # Apply preprocessing logic (same as in main.py)
    df["sex_binary"] = np.where(df["sex"] == 1, 1, 0)
    df["occupation_binary"] = np.where(df["occupation"] >= 500, 1, 0)

    # Verify binary attributes
    assert set(df["sex_binary"].unique()) == {0, 1}
    assert set(df["occupation_binary"].unique()) == {0, 1}

    # Verify correct mapping
    assert df["sex_binary"].tolist() == [1, 0, 1, 0, 1]
    assert df["occupation_binary"].tolist() == [0, 1, 0, 1, 0]

    print("SUCCESS: Dutch preprocessing correctly creates binary attributes")


if __name__ == "__main__":
    test_binary_validation_passes_with_binary_data()
    test_binary_validation_fails_with_non_binary_sensitive()
    test_binary_validation_fails_with_non_binary_target()
    test_dutch_preprocessing_creates_binary_attributes()
    print("\nAll tests passed!")
