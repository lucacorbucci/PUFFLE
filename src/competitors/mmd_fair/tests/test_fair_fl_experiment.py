# ABOUTME: Tests for Fair-FL metric helpers used in MMD-Fair client.
# ABOUTME: Verifies accuracy and demographic disparity computation.

import pytest
import torch

from competitors.mmd_fair.simulation.client import MMDFairFlowerClient


class TestFairMetrics:
    """Test metric computations used during client evaluation."""

    def test_disparity_max(self):
        """Counter-based disparity is 1.0 when groups have opposite positive rates."""
        # z=1 predicted all 0, z=0 predicted all 1 → |P(Y=1|z=1) - P(Y=1|z=0)| = |0 - 1| = 1
        y_pred = torch.tensor([1.0, 1.0, 0.0, 0.0])
        z = torch.tensor([0.0, 0.0, 1.0, 1.0])

        mask_z = z == 1
        mask_not_z = z == 0
        p_y_z = float((y_pred[mask_z] == 1).sum()) / float(mask_z.sum())
        p_y_not_z = float((y_pred[mask_not_z] == 1).sum()) / float(mask_not_z.sum())
        disparity = abs(p_y_z - p_y_not_z)

        assert disparity == pytest.approx(1.0)

    def test_disparity_zero(self):
        """Counter-based disparity is 0.0 when both groups have equal positive rates."""
        y_pred = torch.tensor([1.0, 1.0, 1.0, 1.0])
        z = torch.tensor([0.0, 0.0, 1.0, 1.0])

        mask_z = z == 1
        mask_not_z = z == 0
        p_y_z = float((y_pred[mask_z] == 1).sum()) / float(mask_z.sum())
        p_y_not_z = float((y_pred[mask_not_z] == 1).sum()) / float(mask_not_z.sum())
        disparity = abs(p_y_z - p_y_not_z)

        assert disparity == pytest.approx(0.0)

    def test_client_evaluate_metric_keys(self):
        """Client evaluate() returns PUFFLE-compatible metric keys."""
        import numpy as np
        from FlowerFLTemplate.Utils.preferences import Preferences
        from torch.utils.data import DataLoader, TensorDataset

        prefs = Preferences(
            num_clients=1,
            num_rounds=1,
            dataset_name="dutch",
            model="LinearClassificationNet",
            num_classes=1,
            in_channels=11,
            batch_size=16,
            lr=0.01,
            momentum=0.0,
            weight_decay=0.0,
            regularization_lambda=0.0,
            num_epochs=1,
            fed_dir="/tmp/test_mmd",
        )

        # Small synthetic dataset (x, z, y)
        x = torch.randn(20, 11)
        z = torch.tensor([0] * 10 + [1] * 10, dtype=torch.float32)
        y = torch.randint(0, 2, (20,)).float()
        dataset = TensorDataset(x, z, y)
        loader = DataLoader(dataset, batch_size=16)

        client = MMDFairFlowerClient(
            partition_id=0,
            preferences=prefs,
            trainloader=loader,
            valloader=loader,
        )

        dummy_params = [np.zeros(p.shape) for p in client.model.model.parameters()]
        _, _, metrics = client.evaluate(dummy_params, {})

        required_keys = {
            "client_id",
            "accuracy",
            "loss",
            "disparity",
            "counter_z",
            "counter_not_z",
            "counter_y_z",
            "counter_y_not_z",
            "dataset_counter_z",
            "dataset_counter_not_z",
            "dataset_counter_y_z",
            "dataset_counter_y_not_z",
        }
        assert required_keys.issubset(metrics.keys()), (
            f"Missing keys: {required_keys - metrics.keys()}"
        )
        # Dataset counters should be 0
        assert metrics["dataset_counter_z"] == 0
        assert metrics["dataset_counter_not_z"] == 0

    def test_client_fit_metric_keys(self):
        """Client fit() returns PUFFLE-compatible metric keys including Pk_A0."""
        import numpy as np
        from FlowerFLTemplate.Utils.preferences import Preferences
        from torch.utils.data import DataLoader, TensorDataset

        prefs = Preferences(
            num_clients=1,
            num_rounds=1,
            dataset_name="dutch",
            model="LinearClassificationNet",
            num_classes=1,
            in_channels=11,
            batch_size=16,
            lr=0.01,
            momentum=0.0,
            weight_decay=0.0,
            regularization_lambda=0.0,
            num_epochs=1,
            fed_dir="/tmp/test_mmd",
        )

        x = torch.randn(20, 11)
        z = torch.tensor([0] * 10 + [1] * 10, dtype=torch.float32)
        y = torch.randint(0, 2, (20,)).float()
        dataset = TensorDataset(x, z, y)
        loader = DataLoader(dataset, batch_size=16)

        client = MMDFairFlowerClient(
            partition_id=0,
            preferences=prefs,
            trainloader=loader,
            valloader=loader,
        )

        dummy_params = [np.zeros(p.shape) for p in client.model.model.parameters()]
        _, _, metrics = client.fit(dummy_params, {})

        assert "Pk_A0" in metrics
        assert 0.0 <= metrics["Pk_A0"] <= 1.0
        assert "client_id" in metrics
        assert "lambda" in metrics
        # Training counters from PUFFLEModel
        assert "counter_z" in metrics
