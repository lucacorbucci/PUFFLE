"""
Tests for the FairnessPartitioner class.
"""

import numpy as np
import pandas as pd
from datasets import Dataset

from FlowerFLTemplate.Datasets.Partitioner.fairness_partitioner import (
    FairnessPartitioner,
)

# ruff: noqa: S101, PLR2004


# Helper to create a dummy dataset
def create_dummy_dataset(num_samples=1000):
    """Create a dummy dataset for testing."""
    # 4 groups: (0,0), (0,1), (1,0), (1,1)
    # Balanced-ish
    rng = np.random.default_rng(42)
    data = {
        "target": rng.choice([0, 1], size=num_samples),
        "sensitive": rng.choice([0, 1], size=num_samples),
        "feature": rng.random(num_samples),
    }
    df = pd.DataFrame(data)
    return Dataset.from_pandas(df)


class TestFairnessPartitioner:
    """Test suite for FairnessPartitioner."""

    def test_initialization(self):
        """Test that the partitioner initializes correctly."""
        dataset = create_dummy_dataset(100)
        partitioner = FairnessPartitioner(
            num_partitions=2,
            sensitive_attribute="sensitive",
            target_attribute="target",
            ratio_unfair_clients=0.0,  # All fair
            group_to_reduce=(1, 0),
            ratio_unfairness=(0.5, 0.5),
            group_to_increment=(1, 1),
            dataset=dataset,
        )
        assert partitioner.num_partitions == 2
        assert partitioner.dataset is not None

    def test_fair_nodes_balance(self):
        """Test that fair nodes receive balanced data."""
        # Create a dataset with KNOWN counts to verify math exactly
        # 400 samples, 100 of each group
        df = pd.DataFrame(
            {
                "target": [0] * 100 + [0] * 100 + [1] * 100 + [1] * 100,
                "sensitive": [0] * 100 + [1] * 100 + [0] * 100 + [1] * 100,
            }
        )
        dataset = Dataset.from_pandas(df)

        # 2 partitions, both fair (ratio_unfair=0)
        partitioner = FairnessPartitioner(
            num_partitions=2,
            sensitive_attribute="sensitive",
            target_attribute="target",
            ratio_unfair_clients=0.0,
            group_to_reduce=(1, 0),
            ratio_unfairness=(0.5, 0.5),
            group_to_increment=(1, 1),
            dataset=dataset,
        )

        # Load partition 0
        ds = partitioner.load_partition(0)
        df_part = ds.to_pandas()

        # Should have 50 of each group (100 total / 2)
        counts = df_part.groupby(["target", "sensitive"]).size()
        assert len(counts) == 4
        for count in counts:
            assert count == 50

    def test_unfair_nodes_bias(self):
        """Test that unfair nodes are correctly biased."""
        # 1000 samples, 2 partitions, 50% unfair (1 fair, 1 unfair)
        # Groups: (0,0):250, (0,1):250, (1,0):250, (1,1):250
        df = pd.DataFrame(
            {
                "target": [0] * 250 + [0] * 250 + [1] * 250 + [1] * 250,
                "sensitive": [0] * 250 + [1] * 250 + [0] * 250 + [1] * 250,
            }
        )
        dataset = Dataset.from_pandas(df)

        ratio_reduce = 0.2  # Remove 20%

        partitioner = FairnessPartitioner(
            num_partitions=2,
            sensitive_attribute="sensitive",
            target_attribute="target",
            ratio_unfair_clients=0.5,
            group_to_reduce=(1, 0),  # Remove from this
            ratio_unfairness=(ratio_reduce, ratio_reduce),  # Exact 0.2
            group_to_increment=(1, 1),  # Add to this
            dataset=dataset,
        )

        # Client 0 should be Fair
        ds0 = partitioner.load_partition(0)
        df0 = ds0.to_pandas()
        counts0 = df0.groupby(["target", "sensitive"]).size()
        base0 = counts0.iloc[0]
        # Check all counts are equal (balanced)
        for c in counts0:
            assert c == base0, f"Fair node is not balanced! {counts0}"

        # Client 1 should be Unfair
        ds1 = partitioner.load_partition(1)
        df1 = ds1.to_pandas()
        counts1 = df1.groupby(["target", "sensitive"]).size()

        # Reduced group (1,0) should be less than base
        # (1,0) count
        count_reduced = counts1.get((1, 0), 0)
        # Other groups (0,0) should be base
        count_base = counts1.get((0, 0), 0)

        assert count_reduced < count_base, (
            f"Unfair node did not reduce group (1,0). Got {count_reduced} vs base {count_base}"
        )

        # Incremented group (1,1) should be > base
        count_incremented = counts1.get((1, 1), 0)
        assert count_incremented > count_base, (
            f"Unfair node did not increment group (1,1). Got {count_incremented} vs base {count_base}"
        )

    def test_reproducibility(self):
        """Test that the partitioner is reproducible with the same seed."""
        dataset = create_dummy_dataset(100)
        p1 = FairnessPartitioner(
            num_partitions=2,
            sensitive_attribute="sensitive",
            target_attribute="target",
            ratio_unfair_clients=0.5,
            group_to_reduce=(1, 0),
            ratio_unfairness=(0.5, 0.5),
            dataset=dataset,
            seed=42,
        )
        p2 = FairnessPartitioner(
            num_partitions=2,
            sensitive_attribute="sensitive",
            target_attribute="target",
            ratio_unfair_clients=0.5,
            group_to_reduce=(1, 0),
            ratio_unfairness=(0.5, 0.5),
            dataset=dataset,
            seed=42,
        )

        df1 = p1.load_partition(1).to_pandas()
        df2 = p2.load_partition(1).to_pandas()

        pd.testing.assert_frame_equal(df1, df2)

    def test_client_types_metadata(self):
        """Test that client types are correctly recorded."""
        dataset = create_dummy_dataset(100)
        partitioner = FairnessPartitioner(
            num_partitions=10,
            sensitive_attribute="sensitive",
            target_attribute="target",
            ratio_unfair_clients=0.3,  # 3 unfair, 7 fair
            group_to_reduce=(1, 0),
            ratio_unfairness=(0.5, 0.5),
            dataset=dataset,
        )

        assert len(partitioner.client_types) == 10
        fair_count = sum(1 for t in partitioner.client_types.values() if t == "fair")
        unfair_count = sum(
            1 for t in partitioner.client_types.values() if t == "unfair"
        )

        # 10 * 0.3 = 3 unfair. 7 fair.
        assert unfair_count == 3
        assert fair_count == 7
