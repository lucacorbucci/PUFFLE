import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from puffle.PUFFLEModel.puffle_model import PUFFLEModel


class TestLambdaConstraints:
    """Test suite for lambda regularization parameter constraints."""

    def test_lambda_upper_bound_constraint(self):
        """Test that lambda never exceeds 1.0 during updates."""
        # Setup
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        puffle = PUFFLEModel(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            lambda_regularization=0.9,
            target=0.05,  # Low target to encourage lambda increase
            alpha=0.5,  # High alpha for aggressive updates
            tunable_lambda=True,
        )

        # Simulate high unfairness that would push lambda above 1.0
        high_unfairness = 0.8  # Much higher than target

        # Update lambda multiple times
        for _ in range(10):
            puffle.update_lambda(high_unfairness)
            assert puffle.lambda_regularization <= 1.0, (
                f"Lambda exceeded upper bound: {puffle.lambda_regularization}"
            )
            assert puffle.lambda_regularization >= 0.0, (
                f"Lambda below lower bound: {puffle.lambda_regularization}"
            )

    def test_lambda_lower_bound_constraint(self):
        """Test that lambda never goes below 0.0 during updates."""
        # Setup
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        puffle = PUFFLEModel(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            lambda_regularization=0.1,
            target=0.5,  # High target to encourage lambda decrease
            alpha=0.5,  # High alpha for aggressive updates
            tunable_lambda=True,
        )

        # Simulate low unfairness that would push lambda below 0.0
        low_unfairness = 0.01  # Much lower than target

        # Update lambda multiple times
        for _ in range(10):
            puffle.update_lambda(low_unfairness)
            assert puffle.lambda_regularization >= 0.0, (
                f"Lambda below lower bound: {puffle.lambda_regularization}"
            )
            assert puffle.lambda_regularization <= 1.0, (
                f"Lambda exceeded upper bound: {puffle.lambda_regularization}"
            )

    def test_lambda_stays_at_upper_bound(self):
        """Test that lambda stays at 1.0 when updates would exceed it."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        puffle = PUFFLEModel(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            lambda_regularization=0.95,
            target=0.0,
            alpha=1.0,
            tunable_lambda=True,
        )

        # This should push lambda to exactly 1.0
        puffle.update_lambda(unfairness_loss=0.5)
        assert puffle.lambda_regularization == 1.0

        # Further updates should keep it at 1.0
        puffle.update_lambda(unfairness_loss=1.0)
        assert puffle.lambda_regularization == 1.0

    def test_lambda_stays_at_lower_bound(self):
        """Test that lambda stays at 0.0 when updates would go below it."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        puffle = PUFFLEModel(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            lambda_regularization=0.05,
            target=1.0,
            alpha=1.0,
            tunable_lambda=True,
        )

        # This should push lambda to exactly 0.0
        puffle.update_lambda(unfairness_loss=0.0)
        assert puffle.lambda_regularization == 0.0

        # Further updates should keep it at 0.0
        puffle.update_lambda(unfairness_loss=0.0)
        assert puffle.lambda_regularization == 0.0

    def test_lambda_constraint_with_extreme_alpha(self):
        """Test lambda constraints with very large alpha values."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        puffle = PUFFLEModel(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            lambda_regularization=0.5,
            target=0.5,
            alpha=100.0,  # Extremely high alpha
            tunable_lambda=True,
        )

        # Even with extreme alpha, lambda should stay in bounds
        puffle.update_lambda(unfairness_loss=1.0)
        assert 0.0 <= puffle.lambda_regularization <= 1.0

        puffle.update_lambda(unfairness_loss=0.0)
        assert 0.0 <= puffle.lambda_regularization <= 1.0

    def test_lambda_no_update_without_target(self):
        """Test that lambda doesn't update when target is None."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        initial_lambda = 0.5
        puffle = PUFFLEModel(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            lambda_regularization=initial_lambda,
            target=None,  # No target
            alpha=1.0,
            tunable_lambda=True,
        )

        # Lambda should not change
        puffle.update_lambda(unfairness_loss=0.8)
        assert puffle.lambda_regularization == initial_lambda

    def test_lambda_initialization_within_bounds(self):
        """Test that initial lambda is within [0, 1] bounds."""
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        # Test various initial values
        for initial_lambda in [0.0, 0.25, 0.5, 0.75, 1.0]:
            puffle = PUFFLEModel(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                lambda_regularization=initial_lambda,
            )
            assert 0.0 <= puffle.lambda_regularization <= 1.0

    def test_lambda_constraint_during_training(self):
        """Test lambda constraints during actual training loop."""
        # Create simple synthetic dataset
        x = torch.randn(100, 10)
        y = torch.randint(0, 2, (100,))
        z = torch.randint(0, 2, (100,))
        dataset = TensorDataset(x, z, y)
        loader = DataLoader(dataset, batch_size=10)

        # Setup model with tunable lambda
        model = nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        # Custom criterion that returns a simple loss
        class SimpleCriterion(nn.Module):
            def forward(self, inputs, targets):
                outputs, _z, _lambda = inputs
                return nn.functional.cross_entropy(outputs, targets)

        criterion = SimpleCriterion()

        puffle = PUFFLEModel(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            lambda_regularization=0.5,
            target=0.1,
            alpha=0.1,
            tunable_lambda=True,
        )

        # Train for a few epochs
        metrics = puffle.train(loader, epochs=3, verbose=False)

        # Lambda should still be in bounds after training
        assert 0.0 <= puffle.lambda_regularization <= 1.0, (
            f"Lambda out of bounds after training: {puffle.lambda_regularization}"
        )
        # Verify metrics were collected
        assert len(metrics["train_loss"]) == 3
