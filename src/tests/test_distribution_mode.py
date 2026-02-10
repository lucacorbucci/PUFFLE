"""Tests for distribution_mode parameter in FairnessPartitioner."""
# ruff: noqa: S101, PLC0415, PERF401

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


def test_representative_mode_basic():
    """Test that representative mode works and creates variance between clients."""
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
        distribution_mode="representative",
    )

    # Check that each client has data
    for i in range(num_clients):
        partition = partitioner.load_partition(i)
        assert len(partition) > 0, f"Client {i} has no samples"


def test_per_group_mode_deterministic():
    """Test that per_group mode creates identical fair clients."""
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
        distribution_mode="per_group",
    )

    # Get fair clients (first 5)
    fair_client_sizes = []
    for i in range(5):
        partition = partitioner.load_partition(i)
        fair_client_sizes.append(len(partition))

    # All fair clients should have the same size
    assert len(set(fair_client_sizes)) == 1, "Fair clients should all have same size in per_group mode"


def test_representative_mode_variance():
    """Test that representative mode creates variance between fair clients."""
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
        distribution_mode="representative",
    )

    # Get group distributions for fair clients
    fair_client_group_counts = []
    for i in range(5):  # First 5 are fair
        partition = partitioner.load_partition(i).to_pandas()
        # Count samples from group (1,1)
        count_11 = sum(
            1 for _, row in partition.iterrows()
            if row["target"] == 1 and row["sensitive"] == 1
        )
        fair_client_group_counts.append(count_11)

    # There should be some variance (not all identical)
    # In representative mode, random sampling creates natural variance
    assert len(set(fair_client_group_counts)) > 1, "Fair clients should have variance in representative mode"


def test_mode_switching():
    """Test that both modes work with the same dataset."""
    dataset = create_test_dataset(n_samples=1000)
    num_clients = 10

    # Test per_group mode
    partitioner_pg = FairnessPartitioner(
        num_partitions=num_clients,
        sensitive_attribute="sensitive",
        target_attribute="target",
        ratio_unfair_clients=0.5,
        group_to_reduce=(1, 1),
        ratio_unfairness=(0.8, 0.9),
        group_to_increment=(1, 0),
        seed=42,
        dataset=dataset,
        distribution_mode="per_group",
    )

    # Test representative mode
    partitioner_rep = FairnessPartitioner(
        num_partitions=num_clients,
        sensitive_attribute="sensitive",
        target_attribute="target",
        ratio_unfair_clients=0.5,
        group_to_reduce=(1, 1),
        ratio_unfairness=(0.8, 0.9),
        group_to_increment=(1, 0),
        seed=42,
        dataset=dataset,
        distribution_mode="representative",
    )

    # Both should create valid partitions
    for i in range(num_clients):
        assert len(partitioner_pg.load_partition(i)) > 0
        assert len(partitioner_rep.load_partition(i)) > 0


def test_invalid_mode():
    """Test that invalid distribution_mode raises ValueError."""
    dataset = create_test_dataset(n_samples=1000)

    with pytest.raises(ValueError, match="distribution_mode must be"):
        FairnessPartitioner(
            num_partitions=10,
            sensitive_attribute="sensitive",
            target_attribute="target",
            ratio_unfair_clients=0.5,
            group_to_reduce=(1, 1),
            ratio_unfairness=(0.8, 0.9),
            group_to_increment=(1, 0),
            seed=42,
            dataset=dataset,
            distribution_mode="invalid_mode",
        )


def test_representative_with_samples_per_client():
    """Test representative mode with samples_per_client parameter."""
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
        distribution_mode="representative",
    )

    # Check that clients have approximately the requested samples
    for i in range(num_clients):
        partition = partitioner.load_partition(i)
        # Allow wide tolerance due to fairness manipulation
        assert (
            samples_per_client * 0.5 <= len(partition) <= samples_per_client * 1.5
        ), f"Client {i} has {len(partition)} samples"
