import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

from FlowerFLTemplate.Datasets.Partitioner.fairness_partitioner import (
    FairnessPartitioner,
)


class TestFairnessPartitionerCoverage:
    @pytest.fixture
    def dataframe(self):
        """Create a dummy dataframe for testing."""
        num_samples = 100
        rng = np.random.default_rng(42)
        target = rng.integers(0, 2, num_samples)
        sensitive = rng.integers(0, 2, num_samples)
        return pd.DataFrame(
            {"target": target, "sensitive": sensitive, "data": range(num_samples)}
        )

    @pytest.fixture
    def dataset(self, dataframe):
        """Create a dummy dataset from the dataframe."""
        return Dataset.from_pandas(dataframe)

    def test_partition_none_dataset(self):
        """Test initializing with None dataset doesn't crash and returns empty partitions."""
        partitioner = FairnessPartitioner(
            num_partitions=2,
            sensitive_attribute="sensitive",
            target_attribute="target",
            ratio_unfair_clients=0.5,
            group_to_reduce=(0, 0),
            ratio_unfairness=(0.8, 1.0),
            dataset=None,
        )
        # Should be empty
        assert not partitioner._partitions  # noqa: S101, SLF001

        # Test setter
        # No error should occur if we don't set it yet, but load_partition should fail
        with pytest.raises(ValueError, match="Partition 0 not found"):
            partitioner.load_partition(0)

    def test_dataset_setter(self, dataset):
        """Test setting dataset property triggers partitioning."""
        partitioner = FairnessPartitioner(
            num_partitions=2,
            sensitive_attribute="sensitive",
            target_attribute="target",
            ratio_unfair_clients=0.5,
            group_to_reduce=(0, 0),
            ratio_unfairness=(0.8, 1.0),
            dataset=None,
        )

        partitioner.dataset = dataset
        assert partitioner.dataset is dataset  # noqa: S101
        assert len(partitioner._partitions) == 2  # noqa: S101, SLF001, PLR2004

    def test_group_to_increment_logic(self, dataset):
        """Test the logic where group_to_increment is provided."""
        # Force a case where group_to_increment logic is triggered
        # (1, 1) is a group that exists
        partitioner = FairnessPartitioner(
            num_partitions=2,
            sensitive_attribute="sensitive",
            target_attribute="target",
            ratio_unfair_clients=0.5,  # 1 unfair, 1 fair
            group_to_reduce=(0, 0),
            ratio_unfairness=(1.0, 1.0),  # Force removal
            group_to_increment=(1, 1),
            dataset=dataset,
        )

        # Verify partitioning happened
        assert len(partitioner._partitions) == 2  # noqa: S101, SLF001, PLR2004
        # Check unfair node (index 1) has been processed
        # Hard to verify exact content without complex math, but verifying no crash
        # and that partitions exist covers the lines.
        p1 = partitioner.load_partition(1)
        assert len(p1) > 0  # noqa: S101

    def test_load_invalid_partition(self, dataset):
        """ "Test loading a non-existent partition raises ValueError."""
        partitioner = FairnessPartitioner(
            num_partitions=2,
            sensitive_attribute="sensitive",
            target_attribute="target",
            ratio_unfair_clients=0.5,
            group_to_reduce=(0, 0),
            ratio_unfairness=(0.8, 1.0),
            dataset=dataset,
        )
        with pytest.raises(ValueError, match="Partition 99 not found"):
            partitioner.load_partition(99)

    def test_increment_pool_exhausted(self):
        """Test case where increment pool is smaller than needed."""
        # Create small dataset
        # Group (0,0) -> 10 samples
        # Group (1,1) -> 2 samples.
        # Partitions = 2. Unfair ratio = 0.5 (1 unfair). Max ratio = 1.0.
        # Group to increment = (1, 1).
        # Denom = 2 + 1*1 = 3. Base = 2 // 3 = 0.
        # Samples per group = 0? That would mean empty partitions for that group.
        # Let's try to make it so base > 0 but pool is small.

        df = pd.DataFrame(
            [{"target": 1, "sensitive": 1, "data": i} for i in range(5)]
            + [{"target": 0, "sensitive": 0, "data": i} for i in range(20)]
        )
        dataset = Dataset.from_pandas(df)

        partitioner = FairnessPartitioner(
            num_partitions=2,
            sensitive_attribute="sensitive",
            target_attribute="target",
            ratio_unfair_clients=0.5,  # 1 unfair (node 1)
            group_to_reduce=(0, 0),
            ratio_unfairness=(1.0, 1.0),  # Remove all (0,0) from unfair node
            group_to_increment=(1, 1),
            dataset=dataset,
        )
        # Should run without error
        assert len(partitioner._partitions) == 2  # noqa: S101, SLF001, PLR2004

    def test_increment_group_not_in_data(self):
        """Test if group_to_increment does not exist in dataset."""
        df = pd.DataFrame([{"target": 0, "sensitive": 0, "data": i} for i in range(20)])
        dataset = Dataset.from_pandas(df)

        partitioner = FairnessPartitioner(
            num_partitions=2,
            sensitive_attribute="sensitive",
            target_attribute="target",
            ratio_unfair_clients=0.5,
            group_to_reduce=(0, 0),
            ratio_unfairness=(0.5, 0.5),
            group_to_increment=(1, 1),  # Doesn't exist
            dataset=dataset,
        )
        assert len(partitioner._partitions) == 2  # noqa: S101, SLF001, PLR2004
