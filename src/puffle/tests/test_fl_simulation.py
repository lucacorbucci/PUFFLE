import torch

from puffle.Utils.metric_utils import compute_binary_statistics


class TestFLSimulation:
    def test_fl_aggregation_disjoint_z(self):
        """
        Simulate FL aggregation with 2 clients having disjoint Z distributions.
        Client A: Only Z=0 data.
        Client B: Only Z=1 data.

        The aggregator should receive valid partial counts from both and reconstruct
        the correct global disparity.
        """
        # Global Data (What we want to achieve)
        # Z=0: 4 samples (2 pos, 2 neg) -> P(Y=1|Z=0) = 0.5
        # Z=1: 4 samples (3 pos, 1 neg) -> P(Y=1|Z=1) = 0.75
        # Expected Disparity = |0.75 - 0.5| = 0.25

        # Client A (Z=0 only)
        z_a = torch.tensor([0, 0, 0, 0])
        y_a = torch.tensor([1, 1, 0, 0])

        # Client B (Z=1 only)
        z_b = torch.tensor([1, 1, 1, 1])
        y_b = torch.tensor([1, 1, 1, 0])

        # Helper to mimic metric.py preparation
        def prepare_inputs(z, y):
            unique_z, z_inverse = torch.unique(z, return_inverse=True)
            unique_y, y_inverse = torch.unique(y, return_inverse=True)
            num_z = len(unique_z)
            num_y = len(unique_y)

            pair_indices = z_inverse * num_y + y_inverse
            pair_counts = torch.bincount(pair_indices, minlength=num_z * num_y).float()
            pair_counts = pair_counts.view(num_z, num_y)

            z_counts = torch.bincount(z_inverse, minlength=num_z).float()

            return num_z, unique_z, unique_y, z_counts, pair_counts

        # --- Client Local Computation ---

        # Client A Preparation
        num_z_a, unique_z_a, unique_y_a, z_counts_a, pair_counts_a = prepare_inputs(
            z_a, y_a
        )

        stats_a = compute_binary_statistics(
            num_z=num_z_a,
            unique_z=unique_z_a,
            unique_y=unique_y_a,
            z_counts=z_counts_a,
            pair_counts=pair_counts_a,
            total_samples=len(z_a),
            z=z_a,
            y=y_a,
        )

        # Client B Preparation
        num_z_b, unique_z_b, unique_y_b, z_counts_b, pair_counts_b = prepare_inputs(
            z_b, y_b
        )

        stats_b = compute_binary_statistics(
            num_z=num_z_b,
            unique_z=unique_z_b,
            unique_y=unique_y_b,
            z_counts=z_counts_b,
            pair_counts=pair_counts_b,
            total_samples=len(z_b),
            z=z_b,
            y=y_b,
        )

        # --- Aggregation ---

        # Sum counters (Client A + Client B)
        agg_stats = {}
        for k in ["counter_z", "counter_not_z", "counter_y_z", "counter_y_not_z"]:
            agg_stats[k] = stats_a.get(k, 0) + stats_b.get(k, 0)

        # Global Counts
        # A contributes 4 to Z=0. B contributes 4 to Z=1.
        # counter_z (Z=1) should be 4.
        # counter_not_z (Z=0) should be 4.

        total_z = agg_stats["counter_z"]
        total_not_z = agg_stats["counter_not_z"]

        assert total_z == 4, f"Expected 4 Z=1 samples, got {total_z}"
        assert total_not_z == 4, f"Expected 4 Z=0 samples, got {total_not_z}"

        # Probabilities
        prob_y_given_z = agg_stats["counter_y_z"] / total_z if total_z > 0 else 0
        prob_y_given_not_z = (
            agg_stats["counter_y_not_z"] / total_not_z if total_not_z > 0 else 0
        )

        disparity = abs(prob_y_given_z - prob_y_given_not_z)

        assert disparity == 0.25, f"Expected 0.25 disparity, got {disparity}"
