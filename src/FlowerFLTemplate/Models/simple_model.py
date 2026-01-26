import torch
from torch import nn
from torch.utils.data import DataLoader

from Models.model import Model


class SimpleModel(Model):
    """
    Concrete implementation of Model for classification tasks.

    Handles training and evaluation with CrossEntropyLoss, computes accuracy.
    """

    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device
    ) -> None:
        """
        Initializes the SimpleModel wrapper for PyTorch models.

        Moves model to device; optimizer and criterion are stored for training/evaluation.

        Args:
            model (nn.Module): The PyTorch model to wrap.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
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
        Trains the classification model on training data for specified epochs.

        Uses CrossEntropyLoss, computes loss and accuracy; trains for 1 epoch (epochs param unused). Note: criterion overridden internally.

        Args:
            trainloader (DataLoader): DataLoader for training data with (sample, _, label) tuples.
            epochs (int): Number of epochs (unused in current implementation).

        Returns:
            dict[str, float]: Dictionary with "loss": average training loss, "accuracy": training accuracy.

        Raises:
            RuntimeError: If training fails due to device or tensor issues.
        """
        # Initialize tracking metrics
        criterion = torch.nn.CrossEntropyLoss()
        self.model.to(self.device)
        self.model.train()
        losses = 0.0
        correct = 0
        for sample, _, label in trainloader:
            sample, labels = sample.to(self.device), label.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(sample)
            loss = criterion(output, labels.long())
            loss.backward()
            self.optimizer.step()
            losses += loss.item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum().item()

        loss = torch.tensor(losses / len(trainloader), device=self.device)
        accuracy = correct / len(trainloader.dataset)  # type: ignore
        return {"loss": loss.item(), "accuracy": accuracy}

    def evaluate(self, testloader: DataLoader) -> dict[str, float]:
        """
        Evaluates the classification model on test data, computing loss and accuracy.

        No gradients; uses CrossEntropyLoss, computes accuracy over dataset. Note: criterion overridden internally.

        Args:
            testloader (DataLoader): DataLoader for test data with (sample, _, label) tuples.

        Returns:
            dict[str, float]: Dictionary with "loss": average loss, "accuracy": test accuracy.

        Raises:
            RuntimeError: If evaluation fails due to device or tensor issues.
        """
        self.model.to(self.device)
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        correct, loss = 0, 0.0
        losses = 0.0

        with torch.no_grad():
            for sample, _, label in testloader:
                images, labels = sample.to(self.device), label.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels.long())
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                losses += loss.item()

        loss = torch.tensor(losses / len(testloader), device=self.device)
        accuracy = correct / len(testloader.dataset)  # type: ignore
        return {"loss": loss.item(), "accuracy": accuracy}
