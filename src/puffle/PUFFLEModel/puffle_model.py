# ABOUTME: Implements the PUFFLEModel wrapper for fairness-aware training.
# ABOUTME: Handles the training loop, evaluation, and logging for the PUFFLE library.

from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from puffle.Utils.lambda_updater import LambdaUpdater, LambdaUpdateStrategy
from puffle.Utils.metric import compute_demographic_disparity


class TrainingBatchResult(NamedTuple):
    loss: float
    correct: int
    total: int
    y_batch: torch.Tensor
    predicted: torch.Tensor
    z_batch: torch.Tensor | list
    unfairness: float


class PUFFLEModel:
    def _initialize_metrics_dict(self) -> dict[str, list]:
        """Initialize tracking metrics dictionary."""
        return {
            "train_loss": [],
            "train_accuracy": [],
            "train_f1": [],
            "train_disparity": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "val_disparity": [],
            "test_loss": [],
            "test_accuracy": [],
            "test_f1": [],
            "test_disparity": [],
        }

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: nn.Module | None = None,
        device: torch.device | str = "cpu",
        lambda_regularization: float = 0.0,
        wandb_run=None,
        target: float | None = None,
        momentum: float = 0.9,
        alpha: float = 0.01,
        weight_decay_alpha: float = 0.99,
        *,
        tunable_lambda: bool = False,
        lambda_update_strategy: LambdaUpdateStrategy
        | str = LambdaUpdateStrategy.GRADIENT,
        # PID-specific parameters (only used if strategy is PID)
        lambda_kp: float = 0.01,
        lambda_ki: float = 0.001,
        lambda_kd: float = 0.005,
    ) -> None:
        """
        Initialization of the PUFFLE model.

        Args:
            model (nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer, optional): The optimizer to be used. Defaults to None.
            criterion (nn.Module, optional): The criterion to be used. Defaults to None.
            device (torch.device | str, optional): The device. Defaults to "cpu".
            lambda_regularization (float, optional): The lambda regularization parameter. Defaults to 0.0.
            wandb_run (Any, optional): WandB run for logging. Defaults to None.
            target (float, optional): Target for tunable lambda. Defaults to None.
            momentum (float, optional): Momentum for tunable lambda (momentum strategy). Defaults to 0.9.
            alpha (float, optional): Alpha parameter for tunable lambda (gradient/momentum). Defaults to 0.01.
            weight_decay_alpha (float, optional): Weight decay for alpha. Defaults to 0.99.
            tunable_lambda (bool): Whether to use a tunable lambda.
            lambda_update_strategy (LambdaUpdateStrategy | str): Strategy for lambda updates.
                Options: "momentum", "gradient", "pid". Defaults to "gradient".
            lambda_kp (float): Proportional gain for PID controller. Defaults to 0.01.
            lambda_ki (float): Integral gain for PID controller. Defaults to 0.001.
            lambda_kd (float): Derivative gain for PID controller. Defaults to 0.005.

        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device)
        self.lambda_regularization = lambda_regularization
        self.wandb_run = wandb_run
        self.target = target
        self.momentum = momentum
        self.alpha = alpha
        self.weight_decay_alpha = weight_decay_alpha
        self.tunable_lambda = tunable_lambda
        self.fairness_weight = (
            lambda_regularization  # For backward compatibility if needed
        )
        self.fairness_regularizer = True if lambda_regularization > 0 else None

        # Initialize lambda updater with selected strategy
        if isinstance(lambda_update_strategy, str):
            lambda_update_strategy = LambdaUpdateStrategy(lambda_update_strategy)

        self.lambda_updater = LambdaUpdater(
            strategy=lambda_update_strategy,
            alpha=alpha,
            momentum=momentum,
            kp=lambda_kp,
            ki=lambda_ki,
            kd=lambda_kd,
        )

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the classes for the input features.

        Args:
            x (torch.Tensor): The input features.

        Returns:
            torch.Tensor: The predicted classes.

        """
        self.model.eval()
        self.model = self.model.to(self.device)
        with torch.no_grad():
            outputs = self.model(x.to(self.device))
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu()

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the probabilities for the input features.

        Args:
            x (torch.Tensor): The input features.

        Returns:
            torch.Tensor: The predicted probabilities.

        """
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x.to(self.device))
            probs = F.softmax(outputs, dim=1)
        return probs.cpu()

    def train(
        self,
        train_loader: DataLoader,
        epochs: int,
        val_loader: DataLoader | None = None,
        test_loader: DataLoader | None = None,
        *,
        verbose: bool = True,
        average_probabilities: dict | None = None,
        max_physical_batch_size: int | None = None,
    ) -> dict[str, list[float]]:
        """
        Train the model.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            epochs (int): Number of training epochs.
            val_loader (DataLoader, optional): DataLoader for validation data.
            test_loader (DataLoader, optional): DataLoader for test data.
            verbose (bool): Whether to print progress during training.
            average_probabilities (dict, optional): FL average probabilities.
            max_physical_batch_size (int): Max batch size for privacy accountancy.

        Returns:
            dict[str, list[float]]: Dictionary of metrics tracked during training

        """
        metrics = self._initialize_metrics_dict()
        statistics = []

        effective_max_physical_batch_size = self._get_effective_batch_size(
            train_loader, max_physical_batch_size
        )
        metrics, statistics = self._execute_training_loop(
            epochs,
            train_loader,
            metrics,
            val_loader,
            test_loader,
            average_probabilities,
            verbose,
            effective_max_physical_batch_size,
        )

        # Final conversion of tensor metrics to floats
        for key, value in metrics.items():
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, torch.Tensor):
                        metrics[key][i] = item.item()

        if statistics:
            metrics["counter_z"] = statistics[-1].get("counter_z", 0)
            metrics["counter_not_z"] = statistics[-1].get("counter_not_z", 0)
            metrics["counter_y_z"] = statistics[-1].get("counter_y_z", 0)
            metrics["counter_y_not_z"] = statistics[-1].get("counter_y_not_z", 0)

        return metrics

    def _get_effective_batch_size(self, train_loader, max_physical_batch_size):
        """Determine the effective maximum physical batch size."""
        if max_physical_batch_size is not None:
            return max_physical_batch_size
        return train_loader.batch_size if train_loader.batch_size is not None else 32

    def _execute_training_loop(
        self,
        epochs,
        train_loader,
        metrics,
        val_loader,
        test_loader,
        average_probabilities,
        verbose,
        max_batch_size,
    ):
        """Execute the training loop, optionally using BatchMemoryManager."""
        use_bmm = max_batch_size is not None and hasattr(
            self.optimizer, "signal_skip_step"
        )

        if not use_bmm:
            return self._run_training_loop(
                epochs,
                train_loader,
                metrics,
                val_loader,
                test_loader,
                average_probabilities,
                verbose=verbose,
            )

        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=max_batch_size,
            optimizer=self.optimizer,
        ) as memory_safe_loader:
            return self._run_training_loop(
                epochs,
                memory_safe_loader,
                metrics,
                val_loader,
                test_loader,
                average_probabilities,
                verbose=verbose,
            )

    def _run_training_loop(
        self,
        epochs: int,
        train_loader: DataLoader,
        metrics: dict,
        val_loader: DataLoader | None,
        test_loader: DataLoader | None,
        average_probabilities: dict | None,
        *,
        verbose: bool,
    ) -> tuple[dict, list]:
        statistics = []
        self.model = self.model.to(self.device)
        for epoch in range(epochs):
            # Training
            train_metrics = self._train_one_epoch(
                train_loader,
                _average_probabilities=average_probabilities,
                current_epoch=epoch,
            )

            # Store and log training metrics
            self._update_metrics_dict(metrics, "train", train_metrics)
            statistics.append(train_metrics.get("statistics", {}))
            self._log_wandb_epoch(train_metrics, epoch, mode="train")

            # Validation and Testing
            self._validate_and_test_epoch(
                epoch, epochs, metrics, val_loader, test_loader, verbose=verbose
            )

        return metrics, statistics

    def _update_metrics_dict(self, metrics: dict, prefix: str, epoch_metrics: dict):
        """Update metrics dictionary with epoch results."""
        for key in ["loss", "accuracy", "f1", "disparity"]:
            metrics[f"{prefix}_{key}"].append(epoch_metrics[key])

    def _log_wandb_epoch(self, epoch_metrics: dict, epoch: int, mode: str = "train"):
        """Log epoch metrics to WandB."""
        if not self.wandb_run:
            return

        log_data = {
            f"{mode}_loss": epoch_metrics["loss"],
            f"{mode}_accuracy": epoch_metrics["accuracy"],
            f"{mode}_f1": epoch_metrics["f1"],
            f"{mode}_disparity": epoch_metrics["disparity"],
            "epoch": epoch + 1,
        }

        if mode == "val" and self.target:
            distance = self.target - epoch_metrics["disparity"]
            penalty = 0 if distance > 0 else -1e10
            log_data["Custom_metric"] = epoch_metrics["accuracy"] + penalty

        self.wandb_run.log(log_data)

    def _validate_and_test_epoch(
        self, epoch, epochs, metrics, val_loader, test_loader, *, verbose: bool
    ):
        """Perform validation and testing for the current epoch."""
        if val_loader:
            val_metrics = self.evaluate(val_loader)
            self._update_metrics_dict(metrics, "val", val_metrics)
            self._log_wandb_epoch(val_metrics, epoch, mode="val")

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train loss: {metrics['train_loss'][-1]:.4f}, "
                    f"Val loss: {val_metrics['loss']:.4f}"
                )

        if test_loader:
            test_metrics = self.evaluate(test_loader)
            self._update_metrics_dict(metrics, "test", test_metrics)
            self._log_wandb_epoch(test_metrics, epoch, mode="test")

    def _train_one_epoch(
        self,
        train_loader: DataLoader,
        *,
        current_epoch: int = 0,
        _average_probabilities: dict | None = None,
        _track_metrics_every_n_batches: int | None = None,
        _optimizer_regularization: torch.optim.Optimizer | None = None,
    ) -> dict[str, float]:
        """
        Train the model for one epoch.

        Args:
            train_loader (DataLoader): DataLoader for training data.
            current_epoch (int): Current epoch number.
            average_probabilities (dict, optional): FL average probabilities.
            track_metrics_every_n_batches (int, optional): Log frequency.
            optimizer_regularization (torch.optim.Optimizer, optional): Regularization optimizer.

        Returns:
            dict[str, float]: Metrics for the epoch.

        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        y_true_list = []
        y_pred_list = []
        sensitive_attributes_list = []

        # Loop through batches
        for _batch_idx, batch in enumerate(train_loader):
            result = self._train_batch(
                batch,
                model=self.model,
                optimizer=self.optimizer,
                criterion=self.criterion,
            )

            total_loss += result.loss
            correct += result.correct
            total += result.total

            # Defer cpu/numpy conversion; store tensors temporarily
            y_true_list.append(result.y_batch.detach().cpu())
            y_pred_list.append(result.predicted.detach().cpu())

            # Handle mixed type for z_batch (list or tensor)
            if isinstance(result.z_batch, torch.Tensor):
                sensitive_attributes_list.append(result.z_batch.detach().cpu())
            else:
                sensitive_attributes_list.append(torch.tensor(result.z_batch))

            if self.tunable_lambda:
                self.update_lambda(result.unfairness)
                if self.wandb_run:
                    self.wandb_run.log({"Lambda": self.lambda_regularization})
        if self.tunable_lambda:
            self.update_alpha(current_epoch=current_epoch)
            if self.wandb_run:
                self.wandb_run.log({"Alpha": self.alpha, "Epoch": current_epoch + 1})

        # Concatenate and convert to numpy once at the end
        y_true = torch.cat(y_true_list).numpy().tolist()
        y_pred = torch.cat(y_pred_list).numpy().tolist()
        sensitive_attributes = torch.cat(sensitive_attributes_list).numpy().tolist()

        # Compute final metrics
        return self._compute_metrics(
            total_loss / len(train_loader),
            correct / total,
            y_true,
            y_pred,
            sensitive_attributes,
        )

    def _train_batch(
        self,
        batch,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> TrainingBatchResult:
        """
        Train the model on a single batch.

        Args:
            batch: The current batch of data.
            model (nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer.
            criterion (nn.Module): The loss function.

        Returns:
            TrainingBatchResult: The results of the training step including
            loss, metrics, and batch data.

        """
        x_batch, z_batch, y_batch = batch[0], batch[1], batch[2]

        # Move to device
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        z_batch = (
            z_batch.to(self.device) if isinstance(z_batch, torch.Tensor) else z_batch
        )

        # Forward pass
        optimizer.zero_grad()
        outputs = model(x_batch)
        if criterion is not None:
            # PUFFLE criterion expects a tuple (outputs, z_batch, lambda)
            loss = criterion(
                (outputs, z_batch, self.lambda_regularization), y_batch.long()
            )
        else:
            msg = "Criterion must be provided for training."
            raise ValueError(msg)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Calculate metrics for the batch
        with torch.no_grad():
            softmax_outputs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(softmax_outputs, 1)
            correct_batch = (predicted == y_batch).sum().item()
            total_batch = y_batch.size(0)

            # Use differentiable metric or argmax metric?
            unfairness_batch, _ = compute_demographic_disparity(
                z=z_batch
                if isinstance(z_batch, torch.Tensor)
                else torch.tensor(z_batch, device=self.device),
                y=predicted,
            )

        return TrainingBatchResult(
            loss=loss.item(),
            correct=correct_batch,
            total=total_batch,
            y_batch=y_batch,
            predicted=predicted,
            z_batch=z_batch,
            unfairness=unfairness_batch,
        )

    def evaluate(
        self, data_loader: DataLoader, *, _is_validation: bool = False
    ) -> dict[str, float]:
        """
        Evaluate the model on a dataset.

        Args:
            data_loader (DataLoader): DataLoader for evaluation data.
            is_validation (bool): Whether evaluation is for validation. Defaults to False.

        Returns:
            dict[str, float]: Evaluation metrics.

        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        sensitive_attributes_list = []

        with torch.no_grad():
            for batch in data_loader:
                x_batch, z_batch, y_batch = batch[0], batch[1], batch[2]
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                z_batch = (
                    z_batch.to(self.device)
                    if isinstance(z_batch, torch.Tensor)
                    else z_batch
                )

                outputs = self.model(x_batch)
                if self.criterion:
                    loss = self.criterion(
                        (outputs, z_batch, self.lambda_regularization), y_batch.long()
                    )
                else:
                    loss = torch.tensor(0.0)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                sensitive_attributes_list.extend(
                    z_batch.cpu().numpy()
                    if isinstance(z_batch, torch.Tensor)
                    else z_batch
                )

        return self._compute_metrics(
            total_loss / len(data_loader),
            correct / total,
            y_true,
            y_pred,
            sensitive_attributes_list,
        )

    def _compute_metrics(
        self,
        loss: float,
        accuracy: float,
        y_true: list,
        y_pred: list,
        sensitive_attributes: list,
    ) -> dict[str, float]:
        """
        Compute all metrics.

        Args:
            loss (float): Loss value.
            accuracy (float): Accuracy value.
            y_true (list): True labels.
            y_pred (list): Predicted labels.
            sensitive_attributes (list): Sensitive attributes.

        Returns:
            dict[str, float]: Dictionary of metrics.

        """
        f1 = f1_score(y_true, y_pred, average="weighted")

        # Convert to tensors for disparity computation
        z_tensor = torch.tensor(sensitive_attributes)
        y_tensor = torch.tensor(y_pred)

        try:
            disparity, statistics = compute_demographic_disparity(z_tensor, y_tensor)
        except ValueError:
            disparity = 0.0
            statistics = {}

        return {
            "loss": loss,
            "accuracy": accuracy,
            "f1": f1,
            "disparity": disparity,
            "statistics": statistics,
        }

    def update_lambda(self, unfairness_loss: float) -> None:
        """Update the lambda parameter using the configured strategy."""
        if self.target is not None:
            self.lambda_regularization = self.lambda_updater.update(
                current_lambda=self.lambda_regularization,
                unfairness=unfairness_loss,
                target=self.target,
            )

    def update_alpha(self, *, current_epoch: int) -> None:  # noqa: ARG002
        """Update the alpha parameter."""
        self.alpha = self.alpha * self.weight_decay_alpha

    @staticmethod
    def exp_lr_scheduler(
        initial_alpha: float, current_epoch: int, decay_rate: float = 0.001
    ) -> float:
        """
        Exponential decay for alpha.

        Args:
            initial_alpha (float): Initial alpha value.
            current_epoch (int): Current epoch or round.
            decay_rate (float): Decay rate.

        Returns:
            float: Decayed alpha value.

        """
        return initial_alpha * np.exp(-decay_rate * current_epoch)

    def save(self, path: str) -> None:
        """Save the model state."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict()
                if self.optimizer
                else None,
                "lambda_regularization": self.lambda_regularization,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load the model state."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if self.optimizer and checkpoint["optimizer_state_dict"]:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.lambda_regularization = checkpoint.get(
            "lambda_regularization", self.lambda_regularization
        )
