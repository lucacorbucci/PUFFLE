# ABOUTME: Core PUFFLE model class for fairness-aware training.
# ABOUTME: Implements the training loop, metrics computation, and lambda updates.

from contextlib import contextmanager
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from puffle.Utils.config import PUFFLEConfig
from puffle.Utils.fairness_metrics import FairnessMetrics
from puffle.Utils.lambda_updater import LambdaUpdater
from puffle.Utils.metric import compute_demographic_disparity
from puffle.Utils.modes import MetricMode
from puffle.Utils.tensor_utils import ensure_tensor
from puffle.Utils.types import DeviceType, MetricsDict
from puffle.Utils.privacy import get_noise


class TrainingBatchResult(NamedTuple):
    loss: float
    correct: int
    total: int
    y_batch: torch.Tensor
    predicted: torch.Tensor
    z_batch: torch.Tensor | list
    unfairness: float


class PUFFLEModel:
    def _initialize_metrics_dict(self) -> MetricsDict:
        """Initialize tracking metrics dictionary."""
        return {
            f"{MetricMode.TRAIN.value}_loss": [],
            f"{MetricMode.TRAIN.value}_accuracy": [],
            f"{MetricMode.TRAIN.value}_f1": [],
            f"{MetricMode.TRAIN.value}_disparity": [],
            f"{MetricMode.VALIDATION.value}_loss": [],
            f"{MetricMode.VALIDATION.value}_accuracy": [],
            f"{MetricMode.VALIDATION.value}_f1": [],
            f"{MetricMode.VALIDATION.value}_disparity": [],
            f"{MetricMode.TEST.value}_loss": [],
            f"{MetricMode.TEST.value}_accuracy": [],
            f"{MetricMode.TEST.value}_f1": [],
            f"{MetricMode.TEST.value}_disparity": [],
        }

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: nn.Module | None = None,
        device: DeviceType = "cpu",
        wandb_run: Any | None = None,
        config: PUFFLEConfig | None = None,
    ) -> None:
        """
        Initialization of the PUFFLE model.

        Args:
            model (nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer, optional): The optimizer to be used. Defaults to None.
            criterion (nn.Module, optional): The criterion to be used. Defaults to None.
            device (torch.device | str, optional): The device. Defaults to "cpu".
            wandb_run (Any, optional): WandB run for logging. Defaults to None.
            config (PUFFLEConfig, optional): Configuration for hyperparameters. Defaults to None (uses defaults).

        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device)
        self.wandb_run = wandb_run

        # Initialize config
        self.config = config or PUFFLEConfig()

        # Initialize internal state from config
        self.lambda_regularization = self.config.lambda_regularization
        self.target = self.config.target
        self.momentum = self.config.momentum
        self.alpha = self.config.alpha
        self.weight_decay_alpha = self.config.weight_decay_alpha
        self.tunable_lambda = self.config.tunable_lambda
        self.lambda_updater = LambdaUpdater(
            strategy=self.config.lambda_update_strategy,
            alpha=self.alpha,
            momentum=self.momentum,
            kp=self.config.lambda_kp,
            ki=self.config.lambda_ki,
            kd=self.config.lambda_kd,
        )

    @contextmanager
    def evaluation_mode(self):
        """Context manager to set model to evaluation mode and disable gradients."""
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            if was_training:
                self.model.train()

    @property
    def fairness_regularizer(self) -> bool | None:
        """
        Check if fairness regularization is active.

        Returns:
            bool | None: True if lambda_regularization > 0, else None.

        """
        return True if self.lambda_regularization > 0 else None

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the classes for the input features.

        Args:
            x (torch.Tensor): The input features.

        Returns:
            torch.Tensor: The predicted classes.

        """
        self.model = self.model.to(self.device)
        with self.evaluation_mode():
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
        with self.evaluation_mode():
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

            
            metrics["counter_y_z_noise"] = statistics[-1].get("counter_y_z", 0) + (get_noise(
                    mechanism_type="gaussian",
                    sigma=self.config.sigma_statistics,
                )
                if self.tunable_lambda and self.config.sigma_statistics else 0)
            metrics["counter_y_not_z_noise"] = statistics[-1].get("counter_y_not_z", 0) + (get_noise(
                    mechanism_type="gaussian",
                    sigma=self.config.sigma_statistics,
                )
                if self.tunable_lambda and self.config.sigma_statistics else 0)


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
            self._update_metrics_dict(metrics, MetricMode.TRAIN, train_metrics)
            statistics.append(train_metrics.get("statistics", {}))
            self._log_wandb_epoch(train_metrics, epoch, mode=MetricMode.TRAIN)

            # Validation and Testing
            self._validate_and_test_epoch(
                epoch, epochs, metrics, val_loader, test_loader, verbose=verbose
            )

        return metrics, statistics

    def _update_metrics_dict(
        self,
        metrics: dict,
        prefix: str | MetricMode,
        epoch_metrics: FairnessMetrics | dict,
    ):
        """Update metrics dictionary with epoch results."""
        # Ensure prefix is string
        prefix_str = prefix.value if isinstance(prefix, MetricMode) else prefix
        for key in ["loss", "accuracy", "f1", "disparity"]:
            metrics[f"{prefix_str}_{key}"].append(epoch_metrics[key])

    def _log_wandb_epoch(
        self, epoch_metrics: dict, epoch: int, mode: MetricMode = MetricMode.TRAIN
    ):
        """Log epoch metrics to WandB."""
        if not self.wandb_run:
            return

        prefix = mode.value
        log_data = {
            f"{prefix}_loss": epoch_metrics["loss"],
            f"{prefix}_accuracy": epoch_metrics["accuracy"],
            f"{prefix}_f1": epoch_metrics["f1"],
            f"{prefix}_disparity": epoch_metrics["disparity"],
            "epoch": epoch + 1,
        }

        if mode == MetricMode.VALIDATION and self.target:
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
            self._update_metrics_dict(metrics, MetricMode.VALIDATION, val_metrics)
            self._log_wandb_epoch(val_metrics, epoch, mode=MetricMode.VALIDATION)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Train loss: {metrics[f'{MetricMode.TRAIN.value}_loss'][-1]:.4f}, "
                    f"Val loss: {val_metrics['loss']:.4f}"
                )

        if test_loader:
            test_metrics = self.evaluate(test_loader)
            self._update_metrics_dict(metrics, MetricMode.TEST, test_metrics)
            self._log_wandb_epoch(test_metrics, epoch, mode=MetricMode.TEST)

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
        x_batch = ensure_tensor(batch[0], self.device, dtype=torch.float32)
        z_batch = ensure_tensor(batch[1], self.device)
        y_batch = ensure_tensor(batch[2], self.device)

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
            # Small batches may only have one unique z value, handle gracefully
            # Here we pass noise parameter to compute_demographic_disparity
            # with differential privacy. This is handling everything for us.
            # We do not need to do anything else, even if we are in FL
            # If the noise is None, it will use the default noise = 0. 
            # There are no other differences after this.

            unfairness_batch, _ = compute_demographic_disparity(
                z=z_batch
                if isinstance(z_batch, torch.Tensor)
                else torch.tensor(z_batch, device=self.device),
                y=predicted,
                sigma_update_lambda=self.config.sigma_update_lambda,
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
    ) -> FairnessMetrics:
        """
        Evaluate the model on a dataset.

        Args:
            data_loader (DataLoader): DataLoader for evaluation data.
            _is_validation (bool): Whether evaluation is for validation. Defaults to False.

        Returns:
            FairnessMetrics: Evaluation metrics.

        """
        total_loss = 0.0
        correct = 0
        total = 0
        y_true_list = []
        y_pred_list = []
        sensitive_attributes_list = []

        with self.evaluation_mode():
            for batch in data_loader:
                x_batch = ensure_tensor(batch[0], self.device, dtype=torch.float32)
                z_batch = ensure_tensor(batch[1], self.device)
                y_batch = ensure_tensor(batch[2], self.device)

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

                y_true_list.append(y_batch.detach().cpu())
                y_pred_list.append(predicted.detach().cpu())
                sensitive_attributes_list.append(z_batch.detach().cpu())

        # Concatenate and convert to numpy once at the end
        y_true = torch.cat(y_true_list).numpy().tolist()
        y_pred = torch.cat(y_pred_list).numpy().tolist()
        sensitive_attributes = torch.cat(sensitive_attributes_list).numpy().tolist()

        return self._compute_metrics(
            total_loss / len(data_loader),
            correct / total,
            y_true,
            y_pred,
            sensitive_attributes,
        )

    def _compute_metrics(
        self,
        loss: float,
        accuracy: float,
        y_true: list,
        y_pred: list,
        sensitive_attributes: list,
    ) -> FairnessMetrics:
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
        z_tensor = ensure_tensor(sensitive_attributes, self.device)
        y_tensor = ensure_tensor(y_pred, self.device)

        try:
            disparity, statistics = compute_demographic_disparity(z_tensor, y_tensor)
        except ValueError:
            # Disparity cannot be computed with less than 2 unique z values
            disparity = 0.0
            statistics = {
                "counter_z": 0,
                "counter_not_z": 0,
                "counter_y_z": 0,
                "counter_y_not_z": 0,
            }

        return FairnessMetrics(
            loss=loss,
            accuracy=accuracy,
            f1=float(f1),
            disparity=float(disparity),
            statistics=statistics,
        )

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
