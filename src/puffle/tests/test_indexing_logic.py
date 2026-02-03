import torch

from puffle.Utils.metric_utils import compute_binary_statistics


class TestIndexingLogic:
    def test_indexing_full_case(self):
        """Test case where both Z=0,1 and Y=0,1 are present."""
        z = torch.tensor([0, 0, 1, 1])
        y = torch.tensor([0, 1, 0, 1])
        # Expected:
        # Z=0, Y=0 (1)
        # Z=0, Y=1 (1)
        # Z=1, Y=0 (1)
        # Z=1, Y=1 (1)

        unique_z, z_inverse = torch.unique(z, return_inverse=True)
        unique_y, y_inverse = torch.unique(y, return_inverse=True)
        num_z = len(unique_z)
        num_y = len(unique_y)
        pair_indices = z_inverse * num_y + y_inverse
        pair_counts = (
            torch.bincount(pair_indices, minlength=num_z * num_y)
            .float()
            .view(num_z, num_y)
        )
        z_counts = torch.bincount(z_inverse, minlength=num_z).float()

        stats = compute_binary_statistics(
            num_z, unique_z, unique_y, z_counts, pair_counts, len(z), z, y
        )

        assert stats["counter_z"] == 2
        assert stats["counter_not_z"] == 2
        assert stats["counter_y_z"] == 1
        assert stats["counter_y_not_z"] == 1

    def test_indexing_partial_z_case(self):
        """Test case where only Z=1 is present (common source of bugs)."""
        z = torch.tensor([1, 1])
        y = torch.tensor([0, 1])
        # Expected:
        # Z=1, Y=0 (1) -> counter_not_y_z
        # Z=1, Y=1 (1) -> counter_y_z
        # Z=0 stats -> 0

        unique_z, z_inverse = torch.unique(z, return_inverse=True)
        unique_y, y_inverse = torch.unique(y, return_inverse=True)
        num_z = len(unique_z)
        num_y = len(unique_y)
        pair_indices = z_inverse * num_y + y_inverse
        pair_counts = (
            torch.bincount(pair_indices, minlength=num_z * num_y)
            .float()
            .view(num_z, num_y)
        )
        z_counts = torch.bincount(z_inverse, minlength=num_z).float()

        stats = compute_binary_statistics(
            num_z, unique_z, unique_y, z_counts, pair_counts, len(z), z, y
        )

        assert stats["counter_z"] == 2
        assert stats["counter_not_z"] == 0
        assert stats["counter_y_z"] == 1
        assert stats["counter_y_not_z"] == 0
        assert stats["counter_not_y_z"] == 1

    def test_indexing_partial_y_case(self):
        """Test case where only Y=0 is present."""
        z = torch.tensor([0, 1])
        y = torch.tensor([0, 0])
        # Expected:
        # Z=0, Y=0 (1) -> counter_not_y_not_z
        # Z=1, Y=0 (1) -> counter_not_y_z
        # Y=1 stats -> 0

        unique_z, z_inverse = torch.unique(z, return_inverse=True)
        unique_y, y_inverse = torch.unique(y, return_inverse=True)
        num_z = len(unique_z)
        num_y = len(unique_y)
        pair_indices = z_inverse * num_y + y_inverse
        pair_counts = (
            torch.bincount(pair_indices, minlength=num_z * num_y)
            .float()
            .view(num_z, num_y)
        )
        z_counts = torch.bincount(z_inverse, minlength=num_z).float()

        stats = compute_binary_statistics(
            num_z, unique_z, unique_y, z_counts, pair_counts, len(z), z, y
        )

        assert stats["counter_y"] == 0
        assert stats["counter_y_z"] == 0
        assert stats["counter_not_y_z"] == 1
        assert stats["counter_not_y_not_z"] == 1

    def test_disparity_calculation_consistency(self):
        """Verify that the counters produce the expected disparity."""
        # Simulated scenario: High bias against Z=1
        # Z=1: 100 samples. 5 positive (Y=1). P(Y=1|Z=1) = 0.05
        # Z=0: 100 samples. 50 positive (Y=1). P(Y=1|Z=0) = 0.50

        z = torch.cat([torch.ones(100), torch.zeros(100)])
        y = torch.cat(
            [
                torch.ones(5),
                torch.zeros(95),  # Z=1
                torch.ones(50),
                torch.zeros(50),  # Z=0
            ]
        )

        unique_z, z_inverse = torch.unique(z, return_inverse=True)
        unique_y, y_inverse = torch.unique(y, return_inverse=True)
        num_z = len(unique_z)
        num_y = len(unique_y)
        pair_indices = z_inverse * num_y + y_inverse
        pair_counts = (
            torch.bincount(pair_indices, minlength=num_z * num_y)
            .float()
            .view(num_z, num_y)
        )
        z_counts = torch.bincount(z_inverse, minlength=num_z).float()

        stats = compute_binary_statistics(
            num_z, unique_z, unique_y, z_counts, pair_counts, len(z), z, y
        )

        assert stats["counter_y_z"] == 5
        assert stats["counter_y_not_z"] == 50

        # Manually calculate disparity from stats
        p_y_z = stats["counter_y_z"] / stats["counter_z"]
        p_y_not_z = stats["counter_y_not_z"] / stats["counter_not_z"]
        disparity = abs(p_y_z - p_y_not_z)

        assert abs(disparity - 0.45) < 1e-5
