import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader

from Models.model import Model


class RegressionModel(Model):
    """
    Concrete implementation of Model for regression tasks.

    Handles training and evaluation with MSE loss, computes additional metrics like RMSE, MAE, R2, MSE.
    """

    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device
    ) -> None:
        """
        Initializes the RegressionModel wrapper for PyTorch models.

        Moves model to device; optimizer and criterion are stored for training/evaluation.

        Args:
            model (nn.Module): The PyTorch model to wrap.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            criterion (nn.Module): Loss function (e.g., MSELoss).
            device (torch.device): Compute device (CPU/GPU).

        Returns:
            None
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)

    def train(
        self,
        trainloader: DataLoader,
        epochs: int,
    ) -> dict[str, float]:
        """
        Trains the regression model on training data for specified epochs.

        Performs forward pass, computes loss, backpropagates, updates weights; averages loss over batches. Note: epochs param unused in loop (trains for 1 epoch).

        Args:
            trainloader (DataLoader): DataLoader for training data with (sample, _, label) tuples.
            epochs (int): Number of epochs (unused in current implementation).

        Returns:
            dict[str, float]: Dictionary with "loss": average training loss.

        Raises:
            RuntimeError: If training fails due to device or tensor issues.
        """
        # Initialize tracking metrics

        self.model.to(self.device)
        self.model.train()
        losses = 0.0
        for sample, _, label in trainloader:
            sample, labels = sample.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(sample)
            loss = self.criterion(output, labels)
            loss.backward()
            self.optimizer.step()
            losses += loss.item()

        loss = torch.tensor(losses / len(trainloader), device=self.device)
        return {"loss": loss.item()}

    def evaluate(self, testloader: DataLoader) -> dict[str, float]:
        """
        Evaluates the regression model on test data, computing loss and sklearn metrics.

        No gradients; collects predictions/actuals for MSE, MAE, R2, RMSE; averages loss over batches.

        Args:
            testloader (DataLoader): DataLoader for test data with (sample, _, label) tuples.

        Returns:
            dict[str, float]: Dictionary with "loss": average loss, "rmse", "mae", "r2", "mse".

        Raises:
            RuntimeError: If evaluation fails due to device or tensor issues.
        """
        self.model.to(self.device)
        self.model.eval()
        losses = 0.0

        predictions = []
        actuals = []

        with torch.no_grad():
            for sample, _, label in testloader:
                images, labels = sample.to(self.device), label.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                losses += loss.item()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(label.numpy())

        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()

        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        rmse = np.sqrt(mse)

        loss = torch.tensor(losses / len(testloader), device=self.device)

        return {"loss": loss.item(), "rmse": rmse, "mae": mae, "r2": r2, "mse": mse}
