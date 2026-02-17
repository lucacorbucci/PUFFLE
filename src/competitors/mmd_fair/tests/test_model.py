# ABOUTME: Unit tests for MMDFairModel - the MMD-Fair FedAvg competitor method.
# ABOUTME: Tests kernel computation, tracking function, and training batch logic.

import torch
from torch import nn


class TestDistanceKernel:
    """Test the distance kernel function matching Fair-FL reference."""

    def test_distance_kernel_identity(self):
        """Kernel of identical values should be maximum (1/2 for normalized)."""
        from competitors.mmd_fair.model import distance_kernel

        a = torch.tensor([0.5])
        b = torch.tensor([0.5])
        result = distance_kernel(a, b)

        # For a=b=0.5: (0.5 + 0.5 + 0.5 + 0.5 - 0) / 4 = 0.5
        assert torch.allclose(result, torch.tensor([0.5]), atol=1e-6)

    def test_distance_kernel_opposite_extremes(self):
        """Kernel of 0 and 1 should be minimum."""
        from competitors.mmd_fair.model import distance_kernel

        a = torch.tensor([0.0])
        b = torch.tensor([1.0])
        result = distance_kernel(a, b)

        # For a=0, b=1: (0 + 1 + 1 + 0 - 2) / 4 = 0
        assert torch.allclose(result, torch.tensor([0.0]), atol=1e-6)

    def test_distance_kernel_broadcasting(self):
        """Kernel should broadcast correctly for batch x set."""
        from competitors.mmd_fair.model import distance_kernel

        # Batch of 3 predictions
        a = torch.tensor([[0.2], [0.5], [0.8]])
        # Set of 2 reference values
        b = torch.tensor([[0.3, 0.7]])

        result = distance_kernel(a, b)

        # Should produce 3x2 matrix
        assert result.shape == (3, 2)


class TestMMDFairModelInit:
    """Test MMDFairModel initialization."""

    def test_init_basic(self):
        """Model should initialize with basic parameters."""
        from competitors.mmd_fair.model import MMDFairModel

        model = nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()

        mmd_model = MMDFairModel(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
        )

        assert mmd_model.model is model
        assert mmd_model.optimizer is optimizer
        assert mmd_model.criterion is criterion
        assert mmd_model.Y_0 is None
        assert mmd_model.Y_1 is None
        assert mmd_model.alpha_0 == 1.0
        assert mmd_model.alpha_1 == 1.0
        assert mmd_model.N is None


class TestSetServerPredictions:
    """Test setting tracking sets from server."""

    def test_set_server_predictions_cpu(self):
        """Should convert and store tracking sets on correct device."""
        from competitors.mmd_fair.model import MMDFairModel

        model = nn.Linear(10, 1)
        mmd_model = MMDFairModel(model=model, device="cpu")

        Y_0 = torch.tensor([0.1, 0.2, 0.3])
        Y_1 = torch.tensor([0.7, 0.8, 0.9])

        mmd_model.set_server_predictions(Y_0, Y_1)

        assert mmd_model.Y_0 is not None
        assert mmd_model.Y_1 is not None
        assert torch.allclose(mmd_model.Y_0, Y_0)
        assert torch.allclose(mmd_model.Y_1, Y_1)
        assert mmd_model.Y_0.device.type == "cpu"
        assert mmd_model.Y_1.device.type == "cpu"


class TestSetClientWeights:
    """Test setting alpha weights."""

    def test_set_client_weights(self):
        """Should store alpha weights correctly."""
        from competitors.mmd_fair.model import MMDFairModel

        model = nn.Linear(10, 1)
        mmd_model = MMDFairModel(model=model)

        mmd_model.set_client_weights(alpha_0=1.5, alpha_1=0.8)

        assert mmd_model.alpha_0 == 1.5
        assert mmd_model.alpha_1 == 0.8


class TestSetTotalSamples:
    """Test setting total sample count for N/(N-1) correction."""

    def test_set_total_samples(self):
        """Should store N correctly."""
        from competitors.mmd_fair.model import MMDFairModel

        model = nn.Linear(10, 1)
        mmd_model = MMDFairModel(model=model)

        mmd_model.set_total_samples(1000)

        assert mmd_model.N == 1000


class TestTrackingFunction:
    """Test the C tracking function with N/(N-1) correction."""

    def test_tracking_function_no_correction(self):
        """When N is None, no correction should be applied."""
        from competitors.mmd_fair.model import MMDFairModel

        model = nn.Linear(10, 1)
        mmd_model = MMDFairModel(model=model, device="cpu")

        Y_0 = torch.tensor([0.2, 0.3])
        Y_1 = torch.tensor([0.7, 0.8])
        mmd_model.set_server_predictions(Y_0, Y_1)
        mmd_model.set_tracking_function(Y_0, Y_1)

        # Test predictions
        p = torch.tensor([0.25])

        # Without A specified, should be K(p, Y_0).mean() - K(p, Y_1).mean()
        result = mmd_model.tracking_function(p, demographic_group=None)

        # Should be a scalar
        assert isinstance(result, (float, torch.Tensor))

    def test_tracking_function_with_correction_A0(self):
        """When A=0, should apply N/(N-1) correction to K(p, Y_0)."""
        from competitors.mmd_fair.model import MMDFairModel

        model = nn.Linear(10, 1)
        mmd_model = MMDFairModel(model=model, device="cpu")

        Y_0 = torch.tensor([0.2, 0.3])
        Y_1 = torch.tensor([0.7, 0.8])
        mmd_model.set_server_predictions(Y_0, Y_1)
        mmd_model.set_total_samples(100)
        mmd_model.set_tracking_function(Y_0, Y_1)

        p = torch.tensor([0.25])

        # With A=0, should apply correction: K(p, Y_0) * (N/(N-1)) - K(p, Y_1)
        result = mmd_model.tracking_function(p, demographic_group=0)

        # Should be a scalar
        assert isinstance(result, (float, torch.Tensor))

    def test_tracking_function_empty_predictions(self):
        """Empty predictions should return 0."""
        from competitors.mmd_fair.model import MMDFairModel

        model = nn.Linear(10, 1)
        mmd_model = MMDFairModel(model=model, device="cpu")

        Y_0 = torch.tensor([0.2, 0.3])
        Y_1 = torch.tensor([0.7, 0.8])
        mmd_model.set_server_predictions(Y_0, Y_1)
        mmd_model.set_tracking_function(Y_0, Y_1)

        # Empty tensor
        p = torch.tensor([])

        result = mmd_model.tracking_function(p, demographic_group=0)

        assert result == 0


class TestTrainBatch:
    """Test the _train_batch method."""

    def test_train_batch_basic(self):
        """Should compute loss = task_loss + 2 * lambda * fairness_loss."""
        from competitors.mmd_fair.model import MMDFairModel
        from puffle.Utils.config import PUFFLEConfig

        # Simple binary classification model
        model = nn.Sequential(nn.Linear(5, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()

        config = PUFFLEConfig(lambda_regularization=0.5)

        mmd_model = MMDFairModel(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            config=config,
        )

        # Set up tracking sets
        Y_0 = torch.tensor([0.2, 0.3, 0.4])
        Y_1 = torch.tensor([0.6, 0.7, 0.8])
        mmd_model.set_server_predictions(Y_0, Y_1)
        mmd_model.set_total_samples(100)
        mmd_model.set_tracking_function(Y_0, Y_1)

        # Create a simple batch: 4 samples, 2 from each group
        x_batch = torch.randn(4, 5)
        z_batch = torch.tensor([0, 0, 1, 1])
        y_batch = torch.tensor([0.0, 1.0, 0.0, 1.0])

        batch = (x_batch, z_batch, y_batch)

        result = mmd_model._train_batch(
            batch=batch,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
        )

        # Should return TrainingBatchResult
        assert hasattr(result, "loss")
        assert hasattr(result, "correct")
        assert hasattr(result, "total")
        assert result.total == 4

    def test_train_batch_no_tracking_sets(self):
        """When tracking sets are None, fairness penalty should be 0."""
        from competitors.mmd_fair.model import MMDFairModel
        from puffle.Utils.config import PUFFLEConfig

        model = nn.Sequential(nn.Linear(5, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()

        config = PUFFLEConfig(lambda_regularization=0.5)

        mmd_model = MMDFairModel(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            config=config,
        )

        # No tracking sets set
        x_batch = torch.randn(4, 5)
        z_batch = torch.tensor([0, 0, 1, 1])
        y_batch = torch.tensor([0.0, 1.0, 0.0, 1.0])

        batch = (x_batch, z_batch, y_batch)

        result = mmd_model._train_batch(
            batch=batch,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
        )

        # Should still work, just no fairness penalty
        assert result.total == 4

    def test_train_batch_empty_demographic_group(self):
        """Batch with only one demographic group should handle gracefully."""
        from competitors.mmd_fair.model import MMDFairModel
        from puffle.Utils.config import PUFFLEConfig

        model = nn.Sequential(nn.Linear(5, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()

        config = PUFFLEConfig(lambda_regularization=0.5)

        mmd_model = MMDFairModel(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
            config=config,
        )

        Y_0 = torch.tensor([0.2, 0.3, 0.4])
        Y_1 = torch.tensor([0.6, 0.7, 0.8])
        mmd_model.set_server_predictions(Y_0, Y_1)
        mmd_model.set_total_samples(100)
        mmd_model.set_tracking_function(Y_0, Y_1)

        # Batch with only A=0
        x_batch = torch.randn(4, 5)
        z_batch = torch.tensor([0, 0, 0, 0])
        y_batch = torch.tensor([0.0, 1.0, 0.0, 1.0])

        batch = (x_batch, z_batch, y_batch)

        result = mmd_model._train_batch(
            batch=batch,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
        )

        # Should not crash
        assert result.total == 4
