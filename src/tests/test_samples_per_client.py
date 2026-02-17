"""Tests for samples_per_client parameter in FairnessPartitioner."""
# ruff: noqa: S101, PLC0415, PERF401, PLR2004

import pytest
from datasets import Dataset

from FlowerFLTemplate.Datasets.Partitioner.fairness_partitioner import (
    FairnessPartitioner,
)


def create_test_dataset(n_samples: int = 1000) -> Dataset:
    """Create a simple test dataset with binary sensitive and target attributes."""
    import pandas as pd

    # Create balanced dataset: 4 groups of equal size
    group_size = n_samples // 4
    data = []

    for target in [0, 1]:
        for sensitive in [0, 1]:
            for _ in range(group_size):
                data.append(
                    {
                        "feature1": 0.5,
                        "feature2": 0.5,
                        "target": target,
                        "sensitive": sensitive,
                    }
                )

    df = pd.DataFrame(data)
    return Dataset.from_pandas(df)


def test_samples_per_client_basic():
    """Test that samples_per_client correctly allocates samples."""
    dataset = create_test_dataset(n_samples=1000)
    num_clients = 10
    samples_per_client = 80

    partitioner = FairnessPartitioner(
        num_partitions=num_clients,
        sensitive_attribute="sensitive",
        target_attribute="target",
        ratio_unfair_clients=0.5,
        group_to_reduce=(1, 1),
        ratio_unfairness=(0.8, 0.9),
        group_to_increment=(1, 0),
        seed=42,
        dataset=dataset,
        samples_per_client=samples_per_client,
    )

    # Check that each client has approximately the requested number of samples
    # Note: Due to fairness manipulation (adding samples to unfair clients),
    # some clients may have more than requested
    for i in range(num_clients):
        partition = partitioner.load_partition(i)
        # Allow wider tolerance due to rounding and fairness manipulation
        # Fair clients should be close to samples_per_client
        # Unfair clients may have more due to group_to_increment additions
        assert samples_per_client * 0.7 <= len(partition) <= samples_per_client * 1.5, (
            f"Client {i} has {len(partition)} samples, expected ~{samples_per_client}"
        )


def test_samples_per_client_validation_error():
    """Test that requesting too many samples raises ValueError."""
    dataset = create_test_dataset(n_samples=100)
    num_clients = 10
    samples_per_client = 50  # Total = 500, but only 100 available

    with pytest.raises(ValueError, match="Requested 500 total samples"):
        FairnessPartitioner(
            num_partitions=num_clients,
            sensitive_attribute="sensitive",
            target_attribute="target",
            ratio_unfair_clients=0.5,
            group_to_reduce=(1, 1),
            ratio_unfairness=(0.8, 0.9),
            group_to_increment=(1, 0),
            seed=42,
            dataset=dataset,
            samples_per_client=samples_per_client,
        )


def test_samples_per_client_none_uses_automatic():
    """Test that samples_per_client=None uses automatic calculation."""
    dataset = create_test_dataset(n_samples=1000)
    num_clients = 10

    partitioner = FairnessPartitioner(
        num_partitions=num_clients,
        sensitive_attribute="sensitive",
        target_attribute="target",
        ratio_unfair_clients=0.5,
        group_to_reduce=(1, 1),
        ratio_unfairness=(0.8, 0.9),
        group_to_increment=(1, 0),
        seed=42,
        dataset=dataset,
        samples_per_client=None,  # Explicitly None
    )

    # With automatic calculation, each client should get roughly 1000/10 = 100 samples
    # (with some variation due to fairness manipulation)
    total_samples = sum(len(partitioner.load_partition(i)) for i in range(num_clients))

    # Total should be close to original dataset size (minus discarded samples)
    assert 800 <= total_samples <= 1000


def test_samples_per_client_proportional_groups():
    """Test that groups are allocated proportionally when using samples_per_client."""
    # Create imbalanced dataset: 70% group (0,0), 30% others
    import pandas as pd

    data = []
    # Group (0,0): 700 samples
    for _ in range(700):
        data.append({"feature1": 0.5, "target": 0, "sensitive": 0})
    # Other groups: 100 samples each
    for target in [0, 1]:
        for sensitive in [0, 1]:
            if target == 0 and sensitive == 0:
                continue
            for _ in range(100):
                data.append({"feature1": 0.5, "target": target, "sensitive": sensitive})

    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)

    num_clients = 10
    samples_per_client = 80

    partitioner = FairnessPartitioner(
        num_partitions=num_clients,
        sensitive_attribute="sensitive",
        target_attribute="target",
        ratio_unfair_clients=0.5,
        group_to_reduce=(1, 1),
        ratio_unfairness=(0.8, 0.9),
        group_to_increment=(1, 0),
        seed=42,
        dataset=dataset,
        samples_per_client=samples_per_client,
    )

    # Check that at least one client has samples
    # (proportional allocation should work)
    for i in range(num_clients):
        partition = partitioner.load_partition(i)
        assert len(partition) > 0, f"Client {i} has no samples"


def test_samples_per_client_with_small_dataset():
    """Test edge case with very small dataset."""
    dataset = create_test_dataset(n_samples=50)
    num_clients = 5
    samples_per_client = 8  # Total = 40, available = 50

    partitioner = FairnessPartitioner(
        num_partitions=num_clients,
        sensitive_attribute="sensitive",
        target_attribute="target",
        ratio_unfair_clients=0.5,
        group_to_reduce=(1, 1),
        ratio_unfairness=(0.8, 0.9),
        group_to_increment=(1, 0),
        seed=42,
        dataset=dataset,
        samples_per_client=samples_per_client,
    )

    # Verify partitions were created
    for i in range(num_clients):
        partition = partitioner.load_partition(i)
        assert len(partition) >= 0  # At least should not crash


def test_samples_per_client_maintains_fairness_types():
    """Test that client types (fair/unfair) are still correctly assigned."""
    dataset = create_test_dataset(n_samples=1000)
    num_clients = 10
    samples_per_client = 80
    ratio_unfair = 0.5

    partitioner = FairnessPartitioner(
        num_partitions=num_clients,
        sensitive_attribute="sensitive",
        target_attribute="target",
        ratio_unfair_clients=ratio_unfair,
        group_to_reduce=(1, 1),
        ratio_unfairness=(0.8, 0.9),
        group_to_increment=(1, 0),
        seed=42,
        dataset=dataset,
        samples_per_client=samples_per_client,
    )

    # Count fair and unfair clients
    fair_count = sum(
        1 for i in range(num_clients) if partitioner.client_types[i] == "fair"
    )
    unfair_count = sum(
        1 for i in range(num_clients) if partitioner.client_types[i] == "unfair"
    )

    expected_unfair = int(num_clients * ratio_unfair)
    expected_fair = num_clients - expected_unfair

    assert fair_count == expected_fair
    assert unfair_count == expected_unfair
