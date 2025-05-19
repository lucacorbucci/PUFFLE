import pytest
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from unittest.mock import Mock, patch

from puffle.FairModel.fair_model import PUFFLEModel


# Create a simple model for testing
class SimpleModel(nn.Module):
    def __init__(self, input_dim=2, output_dim=2):
        super(SimpleModel, self).__init__()
        self.layer = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.layer(x)


# Create a simple dataset for testing
class SimpleDataset(Dataset):
    def __init__(self, num_samples=100, input_dim=2, binary_sensitive=True, binary_target=True):
        self.num_samples = num_samples
        
        # Create random data
        self.features = torch.randn(num_samples, input_dim)
        if binary_sensitive:
            self.sensitive_attributes = torch.randint(0, 2, (num_samples,))
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
            idx
        )


class TestPUFFLEModel:
    @pytest.fixture
    def simple_model(self):
        return SimpleModel()
    
    @pytest.fixture
    def simple_dataset(self):
        return SimpleDataset()
    
    @pytest.fixture
    def puffle_model(self, simple_model):
        return PUFFLEModel(
            model=simple_model,
            fairness_weight=0.0,
            seed=42
        )
    
    @pytest.fixture
    def fair_puffle_model(self, simple_model):
        return PUFFLEModel(
            model=simple_model,
            fairness_weight=0.1,  # Non-zero for fairness regularization
            seed=42
        )
    
    def test_initialization(self, simple_model):
        """Test that the model initializes correctly."""
        # Test default initialization
        model = PUFFLEModel(model=simple_model)
        assert model.model is simple_model
        assert model.fairness_weight == 0.0
        assert model.fairness_regularizer is None
        
        # Test with fairness regularization
        model = PUFFLEModel(model=simple_model, fairness_weight=0.1)
        assert model.fairness_weight == 0.1
        assert model.fairness_regularizer is not None
        
        # Test with custom optimizer
        optimizer = torch.optim.SGD(simple_model.parameters(), lr=0.01)
        model = PUFFLEModel(model=simple_model, optimizer=optimizer)
        assert model.optimizer is optimizer
        
        # Test with custom criterion
        criterion = nn.MSELoss()
        model = PUFFLEModel(model=simple_model, criterion=criterion)
        assert model.criterion is criterion
    
    def test_predict(self, puffle_model):
        """Test the predict method."""
        # Create a small batch of data
        x = torch.randn(10, 2)
        
        # Get predictions
        predictions = puffle_model.predict(x)
        
        # Check output shape
        assert predictions.shape == (10,)
        
        # Check output is on CPU
        assert predictions.device.type == 'cpu'
    
    def test_predict_proba(self, puffle_model):
        """Test the predict_proba method."""
        # Create a small batch of data
        x = torch.randn(10, 2)
        
        # Get probability predictions
        probs = puffle_model.predict_proba(x)
        
        # Check output shape
        assert probs.shape == (10, 2)
        
        # Check probabilities sum to 1 for each sample
        row_sums = probs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6)
        
        # Check values are between 0 and 1
        assert (probs >= 0).all().item()
        assert (probs <= 1).all().item()
    
    def test_evaluate(self, puffle_model, simple_dataset):
        """Test evaluation of the model."""
        # Create data loader
        data_loader = DataLoader(simple_dataset, batch_size=32, shuffle=False)
        
        # Evaluate the model
        eval_metrics = puffle_model.evaluate(data_loader)
        
        # Check that metrics were calculated
        assert 'loss' in eval_metrics
        assert 'accuracy' in eval_metrics
        assert 'f1' in eval_metrics
        assert 'disparity' in eval_metrics
        
        # Check types
        assert isinstance(eval_metrics['loss'], float)
        assert isinstance(eval_metrics['accuracy'], float)
        assert isinstance(eval_metrics['f1'], float)
        assert isinstance(eval_metrics['disparity'], float)
        
        # Check value ranges
        assert eval_metrics['loss'] >= 0
        assert 0 <= eval_metrics['accuracy'] <= 1
        assert 0 <= eval_metrics['f1'] <= 1
        assert 0 <= eval_metrics['disparity'] <= 1
    
    def test_compute_metrics(self, puffle_model):
        """Test the _compute_metrics method."""
        loss = 0.5
        accuracy = 0.8
        y_true = [0, 1, 0, 1, 0]
        y_pred = [0, 1, 1, 1, 0]
        sensitive_attributes = [0, 0, 1, 1, 1]
        
        metrics = puffle_model._compute_metrics(loss, accuracy, y_true, y_pred, sensitive_attributes)
        
        # Check that metrics were calculated
        assert metrics['loss'] == loss
        assert metrics['accuracy'] == accuracy
        assert metrics['f1'] > 0  # F1 score should be positive for this test case
        assert 0 <= metrics['disparity'] <= 1  # Disparity should be in [0, 1]
    
    def test_infer_possible_values(self, puffle_model, simple_dataset):
        """Test inferring possible values from a dataset."""
        # Create data loader
        data_loader = DataLoader(simple_dataset, batch_size=32, shuffle=False)
        
        # Infer possible values
        sensitive_attributes, targets = puffle_model._infer_possible_values(data_loader)
        
        # Check that the values were inferred correctly
        assert sorted(sensitive_attributes) == [0, 1]  # Binary sensitive attributes
        assert sorted(targets) == [0, 1]  # Binary targets
    
    @patch('puffle.FairReg.Regularization.RegularizationLoss.RegularizationLoss.__call__')
    def test_train_with_fairness(self, mock_regularizer, fair_puffle_model, simple_dataset):
        """Test training with fairness regularization."""
        # Mock the regularizer to return a known value
        mock_regularizer.return_value = torch.tensor(0.2)
        
        # Create data loaders
        train_loader = DataLoader(simple_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(simple_dataset, batch_size=32, shuffle=False)
        
        # Train the model
        metrics = fair_puffle_model.train(
            train_loader=train_loader,
            epochs=2,
            val_loader=val_loader,
            possible_sensitive_attributes=[0, 1],
            possible_targets=[0, 1],
            verbose=False
        )
        
        # Check that the regularizer was called
        assert mock_regularizer.called
        
        # Check that metrics were tracked
        assert len(metrics['train_loss']) == 2  # 2 epochs
        assert len(metrics['val_loss']) == 2
        
        # Check that metrics have reasonable values
        for metric_list in metrics.values():
            for value in metric_list:
                assert isinstance(value, float)
                assert not np.isnan(value)
                assert not np.isinf(value)
    
    def test_train_without_fairness(self, puffle_model, simple_dataset):
        """Test training without fairness regularization."""
        # Create data loaders
        train_loader = DataLoader(simple_dataset, batch_size=32, shuffle=True)
        
        # Train the model
        metrics = puffle_model.train(
            train_loader=train_loader,
            epochs=2,
            verbose=False
        )
        
        # Check that metrics were tracked
        assert len(metrics['train_loss']) == 2  # 2 epochs
        
        # Check that metrics have reasonable values
        for metric_list in metrics.values():
            if metric_list:  # Skip empty lists (e.g., val_loss when no val_loader)
                for value in metric_list:
                    assert isinstance(value, float)
                    assert not np.isnan(value)
                    assert not np.isinf(value)
    
    def test_early_stopping(self, puffle_model, simple_dataset):
        """Test early stopping functionality."""
        # Mock the evaluate method to return increasing loss values
        original_evaluate = puffle_model.evaluate
        
        # Counter to keep track of evaluate calls
        call_count = [0]
        
        def mock_evaluate(data_loader):
            call_count[0] += 1
            result = {
                'loss': 1.0 + call_count[0] * 0.1,  # Increasing loss
                'accuracy': 0.8,
                'f1': 0.7,
                'disparity': 0.3
            }
            return result
        
        puffle_model.evaluate = mock_evaluate
        
        # Create data loaders
        train_loader = DataLoader(simple_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(simple_dataset, batch_size=32, shuffle=False)
        
        # Train with early stopping (patience=2)
        metrics = puffle_model.train(
            train_loader=train_loader,
            epochs=10,  # Try to train for 10 epochs
            val_loader=val_loader,
            early_stopping_patience=2,
            verbose=False
        )
        
        # Restore original method
        puffle_model.evaluate = original_evaluate
        
        # Check that training was stopped early
        # Should have initial call + patience number of calls
        assert len(metrics['val_loss']) == 3  # Initial + 2 patience calls
    
    def test_save_load(self, puffle_model, simple_model, tmp_path):
        """Test saving and loading the model."""
        # Create a path for the saved model
        model_path = tmp_path / "model.pt"
        
        # Save the model
        puffle_model.save(str(model_path))
        
        # Create a new model with different initial parameters
        new_model = SimpleModel()
        with torch.no_grad():
            # Set parameters to different values
            for param in new_model.parameters():
                param.copy_(torch.randn_like(param))
        
        # Create a new PUFFLE model
        new_puffle_model = PUFFLEModel(model=new_model)
        
        # Load the saved model
        new_puffle_model.load(str(model_path))
        
        # Check that the model parameters are the same
        for p1, p2 in zip(puffle_model.model.parameters(), new_puffle_model.model.parameters()):
            assert torch.allclose(p1, p2)
        
        # Check that the fairness weight is the same
        assert puffle_model.fairness_weight == new_puffle_model.fairness_weight


if __name__ == "__main__":
    pytest.main(["-xvs", "test_fair_model.py"])
