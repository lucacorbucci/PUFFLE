import torch

from puffle.Utils.metric_utils import compute_binary_statistics


class TestMetricUtils:
    """Test suite for metric utilities."""

    def test_binary_statistics_mixed_batches(self):
        """
        Test that counters are semantically consistent across batches with different label distributions.
        Specifically, ensure that 'counter_y_z' always tracks Y=1, even if a batch has only Y=0.
        """
        # Batch A: Standard case (Both classes present)
        # 5 samples: 3 (Z=1, Y=1), 2 (Z=0, Y=0)
        z_a = torch.tensor([1, 1, 1, 0, 0])
        y_a = torch.tensor([1, 1, 1, 0, 0])

        # Pre-compute unique values as the function expects
        unique_z_a = torch.unique(z_a)
        unique_y_a = torch.unique(y_a)
        num_z_a = len(unique_z_a)

        # Bin counts
        z_counts_a = torch.bincount(z_a)
        # Pair counts: Z=0->Y=0 (2), Z=1->Y=1 (3)
        # Flattened indices: 0*2+0=0, 1*2+1=3
        # We need to constructing pair_counts manually to match calling convention
        # pair_counts shape [num_z, num_y]
        pair_counts_a = torch.zeros((2, 2))
        pair_counts_a[0, 0] = 2  # Z=0, Y=0
        pair_counts_a[1, 1] = 3  # Z=1, Y=1

        stats_a = compute_binary_statistics(
            num_z_a,
            unique_z_a,
            unique_y_a,
            z_counts_a,
            pair_counts_a,
            len(z_a),
            z_a,
            y_a,
        )

        # Batch A verification
        assert stats_a["counter_y_z"] == 3, "Batch A: counter_y_z should count Y=1|Z=1"
        assert stats_a["counter_y"] == 3, "Batch A: counter_y should count Y=1"

        # Batch B: Missing positive class (Only Y=0 present)
        # 5 samples: 3 (Z=1, Y=0), 2 (Z=0, Y=0)
        z_b = torch.tensor([1, 1, 1, 0, 0])
        y_b = torch.tensor([0, 0, 0, 0, 0])

        unique_z_b = torch.unique(z_b)
        unique_y_b = torch.unique(y_b)  # [0]
        num_z_b = len(unique_z_b)

        z_counts_b = torch.bincount(z_b)
        # Pair counts: Z=0->Y=0 (2), Z=1->Y=0 (3)
        # Since unique_y has len 1, pair_counts will be [2, 1]
        pair_counts_b = torch.zeros((2, 1))
        pair_counts_b[0, 0] = 2  # Z=0, Y=0
        pair_counts_b[1, 0] = 3  # Z=1, Y=0

        stats_b = compute_binary_statistics(
            num_z_b,
            unique_z_b,
            unique_y_b,
            z_counts_b,
            pair_counts_b,
            len(z_b),
            z_b,
            y_b,
        )

        # Batch B verification - This determines if the bug is present or fixed
        # Expectation: counter_y_z should be 0 because there are NO Y=1 samples.
        # If bug exists, it might be 3 (tracking Y=0|Z=1) or fail.

        # For this test to pass with the FIX, we expect 0.
        # Currently, without the fix, this might assert fail or be non-zero.
        assert stats_b["counter_y_z"] == 0, (
            f"Batch B: counter_y_z should be 0 (no Y=1), got {stats_b.get('counter_y_z')}"
        )
        assert stats_b["counter_y"] == 0, (
            f"Batch B: counter_y should be 0, got {stats_b.get('counter_y')}"
        )
        assert stats_b["counter_not_y_z"] == 3, (
            f"Batch B: counter_not_y_z should be 3, got {stats_b.get('counter_not_y_z')}"
        )
