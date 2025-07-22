from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from puffle.Utils.metric import compute_demographic_disparity


class PUFFLEModel:
    """
    A wrapper for PyTorch models that adds fairness-aware training and evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device | None = None,
    ):
        """
        Initialize the PUFFLEModel wrapper.

        Args:
            model (nn.Module): The PyTorch model to wrap
            optimizer (torch.optim.Optimizer, optional): Optimizer for training. If None,
                                                         Adam will be used with default params.
            criterion (nn.Module): Loss function for training, default is cross entropy
            device (torch.device): Device to use for computation (CPU/GPU)
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        # Set up device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Move model to device
        self.model.to(self.device)

    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True,
        average_probabilities: Optional[Dict] = None,
        max_physical_batch_size: int = 1024,
    ) -> Dict[str, List[float]]:
        """
        Train the model with optional fairness regularization.

        Args:
            train_loader (DataLoader): DataLoader for the training data
            epochs (int): Number of epochs to train for
            val_loader (DataLoader, optional): DataLoader for validation data
            verbose (bool): Whether to print progress during training
            average_probabilities (Dict, optional): Dictionary of probabilities for FL if not all sensitive attributes present

        Returns:
            Dict[str, List[float]]: Dictionary of metrics tracked during training
        """
        # Initialize tracking metrics
        metrics = {
            "train_loss": [],
            "train_accuracy": [],
            "train_f1": [],
            "train_disparity": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_disparity": [],
        }
        
        print(self.criterion)

        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=max_physical_batch_size,
            optimizer=self.optimizer,
        ) as memory_safe_data_loader:
            for epoch in range(epochs):
                # Training
                train_metrics = self._train_one_epoch(
                    memory_safe_data_loader,
                    average_probabilities=average_probabilities,
                )

                # Store metrics
                metrics["train_loss"].append(train_metrics["loss"])
                metrics["train_accuracy"].append(train_metrics["accuracy"])
                metrics["train_f1"].append(train_metrics["f1"])
                metrics["train_disparity"].append(train_metrics["disparity"])

                # Validation if provided
                if val_loader:
                    val_metrics = self.evaluate(val_loader)

                    metrics["val_loss"].append(val_metrics["loss"])
                    metrics["val_accuracy"].append(val_metrics["accuracy"])
                    metrics["val_f1"].append(val_metrics["f1"])
                    metrics["val_disparity"].append(val_metrics["disparity"])

                    if verbose:
                        print(
                            f"Epoch {epoch + 1}/{epochs}, "
                            f"Train Loss: {train_metrics['loss']:.4f}, "
                            f"Train Acc: {train_metrics['accuracy']:.4f}, "
                            f"Train F1: {train_metrics['f1']:.4f}, "
                            f"Train Disparity: {train_metrics['disparity']:.4f}, "
                            f"Val Loss: {val_metrics['loss']:.4f}, "
                            f"Val Acc: {val_metrics['accuracy']:.4f}, "
                            f"Val F1: {val_metrics['f1']:.4f}, "
                            f"Val Disparity: {val_metrics['disparity']:.4f}"
                        )
                else:
                    if verbose:
                        print(
                            f"Epoch {epoch + 1}/{epochs}, "
                            f"Train Loss: {train_metrics['loss']:.4f}, "
                            f"Train Acc: {train_metrics['accuracy']:.4f}, "
                            f"Train F1: {train_metrics['f1']:.4f}, "
                            f"Train Disparity: {train_metrics['disparity']:.4f}"
                        )

        return metrics

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        average_probabilities: Optional[Dict] = None,
        track_metrics_every_n_batches: Optional[int] = None,
        optimizer_regularization: Optional[torch.optim.Optimizer] = None,
    ) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data
            possible_targets (List, optional): List of possible target values
            average_probabilities (Dict, optional): Dictionary of probabilities for FL
            track_metrics_every_n_batches (int, optional): Track metrics every n batches

        Returns:
            Dict[str, float]: Dictionary of metrics for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        sensitive_attributes = []

        # Loop through batches
        for batch_idx, batch in enumerate(train_loader):
            loss_batch, correct_batch, total_batch, y_batch, predicted_batch, z_batch = self._train_batch(
                batch,
                model=self.model,
                optimizer=self.optimizer,
                criterion=self.criterion,
            )

            total_loss += loss_batch
            correct += (predicted_batch == y_batch).sum()
            total += y_batch.size(0)
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted_batch.cpu().numpy())
            sensitive_attributes.extend(z_batch.cpu().numpy() if isinstance(z_batch, torch.Tensor) else z_batch)

        # Compute final metrics
        return self._compute_metrics(
            total_loss / len(train_loader), correct / total, y_true, y_pred, sensitive_attributes
        )

    def _train_batch(
        self,
        batch,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ):
        # Assuming batch contains (x, z, y, _, _) where:
        # x: features, z: sensitive attributes, y: targets, and the last two are indices
        x_batch, z_batch, y_batch = batch[0], batch[1], batch[2]

        # Move to device
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        z_batch = z_batch.to(self.device) if isinstance(z_batch, torch.Tensor) else z_batch

        # Forward pass
        optimizer.zero_grad()
        outputs = model(x_batch)
        if criterion is not None:
            loss = criterion((outputs, z_batch), y_batch.long())
        else:
            raise ValueError("Criterion must be provided for training.")

        # Backward pass and optimize
        loss.backward()
        self.optimizer.step()

        # Calculate metrics
        softmax_outputs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(softmax_outputs, 1)

        correct_batch = (predicted == y_batch).sum().item()
        total_batch = y_batch.size(0)

        return loss.item(), correct_batch, total_batch, y_batch, predicted, z_batch

    def evaluate(self, data_loader: DataLoader, is_validation: bool = False) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.

        Args:
            data_loader (DataLoader): DataLoader for evaluation
            is_validation (bool): Whether this is a validation set

        Returns:
            Dict[str, float]: Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        sensitive_attributes = []

        with torch.no_grad():
            for x_batch, z_batch, y_batch, _, _ in data_loader:
                # Move to device
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                z_batch = z_batch.to(self.device) if isinstance(z_batch, torch.Tensor) else z_batch

                # Forward pass
                outputs = self.model(x_batch)
                loss = self.criterion((outputs, z_batch), y_batch.long())

                # Calculate metrics
                total_loss += loss.item()
                softmax_outputs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(softmax_outputs, 1)
                correct += (predicted == y_batch).sum().item()
                total += y_batch.size(0)

                # Store predictions and ground truth for F1 score and disparity calculation
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                sensitive_attributes.extend(z_batch.cpu().numpy() if isinstance(z_batch, torch.Tensor) else z_batch)

        sensitive_attributes = list(map(int, sensitive_attributes))
        y_true = list(map(int, y_true))
        y_pred = list(map(int, y_pred))

        # Compute metrics
        return self._compute_metrics(
            total_loss / len(data_loader), correct / total, y_true, y_pred, sensitive_attributes
        )

    # def predict(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Make predictions with the model.

    #     Args:
    #         x (torch.Tensor): Input features

    #     Returns:
    #         torch.Tensor: Predicted class labels
    #     """
    #     self.model.eval()
    #     with torch.no_grad():
    #         x = x.to(self.device)
    #         outputs = self.model(x)
    #         softmax_outputs = F.softmax(outputs, dim=1)
    #         _, predicted = torch.max(softmax_outputs, 1)
    #     return predicted

    # def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
    #     """
    #     Get probability predictions with the model.

    #     Args:
    #         x (torch.Tensor): Input features

    #     Returns:
    #         torch.Tensor: Probability predictions
    #     """
    #     self.model.eval()
    #     with torch.no_grad():
    #         x = x.to(self.device)
    #         outputs = self.model(x)
    #         softmax_outputs = F.softmax(outputs, dim=1)
    #     return softmax_outputs

    def _compute_metrics(
        self, loss: float, accuracy: float, y_true: List, y_pred: List, sensitive_attributes: List
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            loss (float): Loss value
            accuracy (float): Accuracy value
            y_true (List): List of ground truth labels
            y_pred (List): List of predicted labels
            sensitive_attributes (List): List of sensitive attributes

        Returns:
            Dict[str, float]: Dictionary of computed metrics
        """
        # Calculate F1 score
        f1 = f1_score(y_true, y_pred, average="macro")

        # Calculate demographic disparity
        disparity = compute_demographic_disparity(z=torch.tensor(sensitive_attributes), y=torch.tensor(y_pred))

        return {"loss": loss, "accuracy": accuracy, "f1": f1, "disparity": disparity}

    # def _infer_possible_values(self, data_loader: DataLoader) -> Tuple[List, List]:
    #     """
    #     Infer possible values for sensitive attributes and targets from the data.

    #     Args:
    #         data_loader (DataLoader): DataLoader to extract values from

    #     Returns:
    #         Tuple[List, List]: Lists of possible sensitive attributes and target values
    #     """
    #     sensitive_attributes = []
    #     targets = []

    #     # Extract unique values from a small portion of the data
    #     for i, (_, z_batch, y_batch, _, _) in enumerate(data_loader):
    #         sensitive_attributes.extend(z_batch.numpy() if isinstance(z_batch, torch.Tensor) else z_batch)
    #         targets.extend(y_batch.numpy() if isinstance(y_batch, torch.Tensor) else y_batch)

    #         # Limit to avoid scanning the entire dataset
    #         if i >= min(10, len(data_loader) - 1):
    #             break

    #     return list(set(sensitive_attributes)), list(set(targets))

    # def save(self, path: str) -> None:
    #     """
    #     Save the model to disk.

    #     Args:
    #         path (str): Path to save the model
    #     """
    #     torch.save(
    #         {
    #             "model_state_dict": self.model.state_dict(),
    #             "optimizer_state_dict": self.optimizer.state_dict(),
    #         },
    #         path,
    #     )

    # def load(self, path: str) -> None:
    #     """
    #     Load the model from disk.

    #     Args:
    #         path (str): Path to load the model from
    #     """
    #     checkpoint = torch.load(path)
    #     self.model.load_state_dict(checkpoint["model_state_dict"])
    #     self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
