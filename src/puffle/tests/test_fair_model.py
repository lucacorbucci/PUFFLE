from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from puffle.PUFFLEModel.puffle_model import PUFFLEModel, TrainingBatchResult


# Create a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=2):
        super().__init__()
        self.layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.layer(x)


# Create a simple dataset for testing
class SimpleDataset(Dataset):
    def __init__(
        self, *, num_samples=100, input_dim=2, binary_sensitive=True, binary_target=True
    ):
        self.num_samples = num_samples

        # Create random data
        self.features = torch.randn(num_samples, input_dim)
        if binary_sensitive:
            # Ensure at least one 0 and one 1 are present
            self.sensitive_attributes = torch.randint(0, 2, (num_samples,))
            self.sensitive_attributes[0] = 0
            self.sensitive_attributes[1] = 1
        else:
            self.sensitive_attributes = torch.randint(0, 3, (num_samples,))

        if binary_target:
            self.targets = torch.randint(0, 2, (num_samples,))
        else:
            self.targets = torch.randint(0, 3, (num_samples,))

        self.indices = torch.arange(num_samples)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.sensitive_attributes[idx].item(),
            self.targets[idx].item(),
            self.indices[idx].item(),
            idx,
        )


class TestPUFFLEModel:
    @pytest.fixture
    def simple_model(self):
        return SimpleModel()

    @pytest.fixture
    def optimizer(self, simple_model):
        return torch.optim.SGD(simple_model.parameters(), lr=0.01)

    @pytest.fixture
    def criterion(self):
        return nn.CrossEntropyLoss()

    @pytest.fixture
    def simple_dataset(self):
        return SimpleDataset()

    @pytest.fixture
    def puffle_model(self, simple_model, optimizer, criterion):
        return PUFFLEModel(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            lambda_regularization=0.0,
        )

    @pytest.fixture
    def fair_puffle_model(self, simple_model, optimizer, criterion):
        return PUFFLEModel(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            lambda_regularization=0.1,
        )

    def test_initialization(self, simple_model, optimizer, criterion):
        """Test that the model initializes correctly."""
        # Test basic initialization
        model = PUFFLEModel(
            model=simple_model, optimizer=optimizer, criterion=criterion
        )
        assert model.model is simple_model
        assert model.optimizer is optimizer
        assert model.criterion is criterion
        assert model.lambda_regularization == 0.0

        # Test with fairness regularization
        model = PUFFLEModel(
            model=simple_model,
            optimizer=optimizer,
            criterion=criterion,
            lambda_regularization=0.1,
        )
        assert model.lambda_regularization == 0.1

    def test_evaluate(self, puffle_model, simple_dataset):
        """Test evaluation of the model."""
        # Create data loader
        data_loader = DataLoader(simple_dataset, batch_size=32, shuffle=False)

        # Mock the criterion to handle the tuple input expected by some fairness losses
        # or just use a standard one and mock the call if needed.
        # Actually, PUFFLEModel.evaluate calls self.criterion((outputs, z_batch, self.lambda_regularization), y_batch.long())

        def mock_criterion(inputs, target):
            outputs, _, _ = inputs
            return nn.CrossEntropyLoss()(outputs, target)

        puffle_model.criterion = mock_criterion

        # Evaluate the model
        eval_metrics = puffle_model.evaluate(data_loader)

        # Check that metrics were calculated
        assert "loss" in eval_metrics
        assert "accuracy" in eval_metrics
        assert "f1" in eval_metrics
        assert "disparity" in eval_metrics

        # Check types
        assert isinstance(eval_metrics["loss"], float)
        assert isinstance(eval_metrics["accuracy"], float)
        assert isinstance(eval_metrics["f1"], float)
        assert isinstance(eval_metrics["disparity"], float)

    def test_compute_metrics(self, puffle_model):
        """Test the _compute_metrics method."""
        loss = 0.5
        accuracy = 0.8
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 1, 1, 0]
        sensitive_attributes = [0, 0, 1, 1, 1]

        metrics = puffle_model._compute_metrics(
            loss, accuracy, y_true, y_pred, sensitive_attributes
        )

        # Check that metrics were calculated
        assert metrics["loss"] == loss
        assert metrics["accuracy"] == accuracy
        assert metrics["f1"] > 0
        assert 0 <= metrics["disparity"] <= 1

    def test_train_without_fairness(self, puffle_model, simple_dataset):
        """Test training without fairness regularization."""
        # Create data loader - use full batch to ensure both sensitive groups are present
        train_loader = DataLoader(simple_dataset, batch_size=100, shuffle=True)

        # Mock criterion
        def mock_criterion(inputs, target):
            outputs, _, _ = inputs
            return nn.CrossEntropyLoss()(outputs, target)

        puffle_model.criterion = mock_criterion

        # Train the model
        metrics = puffle_model.train(train_loader=train_loader, epochs=1, verbose=False)

        # Check that metrics were tracked
        assert len(metrics["train_loss"]) == 1
        assert "train_accuracy" in metrics
        assert "train_disparity" in metrics

    @patch("puffle.PUFFLEModel.puffle_model.BatchMemoryManager")
    def test_train_one_epoch(self, mock_memory_manager, puffle_model, simple_dataset):
        """Test training for one epoch."""
        train_loader = DataLoader(simple_dataset, batch_size=32)
        mock_memory_manager.return_value.__enter__.return_value = train_loader

        # Mock _train_batch
        with patch.object(puffle_model, "_train_batch") as mock_train_batch:
            # Create a mock batch of 32 samples with diverse sensitive attributes
            z_batch = torch.tensor([0, 1] * 16)
            y_batch = torch.randint(0, 2, (32,))
            predicted_batch = torch.randint(0, 2, (32,))

            mock_train_batch.return_value = TrainingBatchResult(
                loss=0.5,
                correct=1,
                total=32,
                y_batch=y_batch,
                predicted=predicted_batch,
                z_batch=z_batch,
                unfairness=0.1,
            )

            puffle_model._train_one_epoch(train_loader)
            assert mock_train_batch.called

    def test_update_lambda(self, puffle_model):
        """Test the update_lambda method."""
        puffle_model.tunable_lambda = True
        puffle_model.target = 0.1
        puffle_model.alpha = 0.01
        puffle_model.lambda_regularization = 0.5

        # Unfairness > target -> lambda should increase
        puffle_model.update_lambda(0.2)
        assert puffle_model.lambda_regularization > 0.5

        # Unfairness < target -> lambda should decrease
        current_lambda = puffle_model.lambda_regularization
        puffle_model.update_lambda(0.05)
        assert puffle_model.lambda_regularization < current_lambda

        # Should not go below 0
        puffle_model.lambda_regularization = 0.0
        puffle_model.update_lambda(0.0)
        assert puffle_model.lambda_regularization == 0.0

    def test_update_alpha(self, puffle_model):
        """Test the update_alpha method."""
        puffle_model.alpha = 0.1
        puffle_model.weight_decay_alpha = 0.9
        puffle_model.update_alpha(current_epoch=1)
        assert puffle_model.alpha == pytest.approx(0.09)

    def test_exp_lr_scheduler(self):
        """Test the exp_lr_scheduler method."""
        initial_alpha = 0.1
        current_epoch = 10
        decay_rate = 0.01
        expected = 0.1 * torch.exp(torch.tensor(-0.01 * 10)).item()
        result = PUFFLEModel.exp_lr_scheduler(initial_alpha, current_epoch, decay_rate)
        assert result == pytest.approx(expected)

    def test_initialize_metrics_dict(self, puffle_model):
        """Test the _initialize_metrics_dict method."""
        metrics = puffle_model._initialize_metrics_dict()
        assert isinstance(metrics, dict)
        assert "train_loss" in metrics
        assert isinstance(metrics["train_loss"], list)
        assert len(metrics["train_loss"]) == 0

    def test_update_metrics_dict(self, puffle_model):
        """Test the _update_metrics_dict method."""
        metrics = puffle_model._initialize_metrics_dict()
        epoch_metrics = {
            "loss": 0.5,
            "accuracy": 0.8,
            "f1": 0.75,
            "disparity": 0.1,
        }
        puffle_model._update_metrics_dict(metrics, "train", epoch_metrics)
        assert metrics["train_loss"] == [0.5]
        assert metrics["train_accuracy"] == [0.8]
        assert metrics["train_f1"] == [0.75]
        assert metrics["train_disparity"] == [0.1]

    def test_get_effective_batch_size(self, puffle_model):
        """Test _get_effective_batch_size."""

        class MockLoader:
            batch_size = 64

        loader = MockLoader()
        # Case 1: max_physical_batch_size is provided
        assert puffle_model._get_effective_batch_size(loader, 32) == 32
        # Case 2: max_physical_batch_size is None, use loader batch_size
        assert puffle_model._get_effective_batch_size(loader, None) == 64
        # Case 3: Both None, use default 32
        loader.batch_size = None
        assert puffle_model._get_effective_batch_size(loader, None) == 32

    def test_train_batch(self, puffle_model):
        """Test the _train_batch method."""
        # Setup mock batch
        x = torch.randn(4, 2)
        z = torch.tensor([0, 1, 0, 1])
        y = torch.tensor([0, 1, 0, 1])
        batch = (x, z, y)

        # Mock criterion
        def mock_criterion(inputs, targets):
            outputs, _, _ = inputs
            return nn.CrossEntropyLoss()(outputs, targets)

        puffle_model.criterion = mock_criterion
        puffle_model.optimizer = torch.optim.Adam(puffle_model.model.parameters())

        # Reset gradients
        puffle_model.optimizer.zero_grad()

        result = puffle_model._train_batch(
            batch,
            model=puffle_model.model,
            optimizer=puffle_model.optimizer,
            criterion=puffle_model.criterion,
        )

        loss, correct, total, y_batch, predicted, z_batch, unfairness = result

        assert isinstance(loss, float)
        assert isinstance(correct, int)
        assert total == 4
        assert torch.equal(y_batch, y.to(puffle_model.device))
        assert predicted.shape == (4,)
        assert torch.equal(z_batch, z.to(puffle_model.device))
        assert isinstance(unfairness, float)

    def test_save_load(self, puffle_model, tmp_path):
        """Test saving and loading the model."""
        save_path = tmp_path / "model.pt"
        puffle_model.lambda_regularization = 0.75
        puffle_model.save(str(save_path))

        # Change lambda and check if load restores it
        puffle_model.lambda_regularization = 0.1
        puffle_model.load(str(save_path))
        assert puffle_model.lambda_regularization == 0.75

    def test_predict(self, puffle_model):
        """Test the predict method."""
        x = torch.randn(5, 2)
        predictions = puffle_model.predict(x)
        assert predictions.shape == (5,)
        assert isinstance(predictions, torch.Tensor)

    def test_predict_proba(self, puffle_model):
        """Test the predict_proba method."""
        x = torch.randn(5, 2)
        probs = puffle_model.predict_proba(x)
        assert probs.shape == (5, 2)
        assert torch.allclose(probs.sum(dim=1), torch.ones(5))


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
