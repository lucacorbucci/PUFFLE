# ABOUTME: MMD-Fair FedAvg model - competitor method for benchmarking.
# ABOUTME: Implements fairness via function tracking with MMD kernel.

import torch
import torch.nn.functional as F
from puffle.PUFFLEModel.puffle_model import PUFFLEModel, TrainingBatchResult
from puffle.Utils.config import PUFFLEConfig
from puffle.Utils.metric import compute_demographic_disparity
from puffle.Utils.tensor_utils import ensure_tensor
from puffle.Utils.types import DeviceType
from torch import nn


def distance_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Distance kernel from Fair-FL reference implementation.

    Formula: ((|a-1| + |b-1| - |a-b|) + (|a| + |b| - |a-b|)) / 4

    For predictions in [0,1], this simplifies to (1 - |a-b|) / 2.

    Args:
        a: Tensor of predictions (batch_size, 1) or (batch_size,)
        b: Tensor of reference values (1, num_samples) or (num_samples,)

    Returns:
        Kernel matrix of shape (batch_size, num_samples)

    """
    # Ensure proper broadcasting shape
    if a.dim() == 1:
        a = a.unsqueeze(1)
    if b.dim() == 1:
        b = b.unsqueeze(0)

    # Ensure both tensors are on the same device
    if a.device != b.device:
        b = b.to(a.device)

    term1 = torch.abs(a - 1) + torch.abs(b - 1) - torch.abs(a - b)
    term2 = torch.abs(a) + torch.abs(b) - torch.abs(a - b)

    return (term1 + term2) / 4


class MMDFairModel(PUFFLEModel):
    """
    MMD-Fair FedAvg model.

    Implements the fairness method from "Global Group Fairness in Federated
    Learning via Function Tracking" (https://arxiv.org/pdf/2503.15163).

    This model subclasses PUFFLEModel to reuse training infrastructure,
    but overrides _train_batch to use the MMD fairness penalty instead of
    the PUFFLE disparity/error-rate regularization.

    Key differences from PUFFLE:
    - Uses a standard task loss (e.g., BCEWithLogitsLoss) instead of MixLoss
    - Computes fairness penalty via tracking function C with kernel K
    - Applies N/(N-1) correction for MMD estimator
    - Alpha weights control sampling proportions, not loss formula
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: nn.Module | None = None,
        device: DeviceType = "cpu",
        wandb_run=None,
        config: PUFFLEConfig | None = None,
    ) -> None:
        """
        Initialize MMDFairModel.

        Args:
            model: The neural network to train
            optimizer: Optimizer (e.g., SGD)
            criterion: Standard task loss (e.g., BCEWithLogitsLoss, CrossEntropyLoss)
            device: Device for computation
            wandb_run: WandB run for logging
            config: PUFFLE config (lambda_regularization controls fairness weight)

        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            wandb_run=wandb_run,
            config=config,
        )

        # MMD tracking state
        self.Y_0: torch.Tensor | None = None
        self.Y_1: torch.Tensor | None = None

        # MMD loss accumulator for metrics
        self.running_mmd_loss: float = 0.0

        # Alpha weights for sampling proportions (not used in loss)
        self.alpha_0: float = 1.0
        self.alpha_1: float = 1.0

        # Total sample count for N/(N-1) correction
        self.N: int | None = None

        # Tracking function closure (set via set_tracking_function)
        def _default_tracking(_p, demographic_group=None):  # noqa: ARG001
            return 0

        self.tracking_function = _default_tracking

    def _initialize_metrics_dict(self):
        """Initialize metrics dict with MMD loss support."""
        metrics = super()._initialize_metrics_dict()
        metrics["train_mmd_loss"] = []
        metrics["validation_mmd_loss"] = []
        metrics["test_mmd_loss"] = []
        return metrics

    def _update_metrics_dict(self, metrics, prefix, epoch_metrics):
        """Update metrics dict with MMD loss."""
        super()._update_metrics_dict(metrics, prefix, epoch_metrics)
        # Handle both dict and FairnessMetrics (via getattr or getitem)
        mmd_loss = (
            epoch_metrics.get("mmd_loss")
            if isinstance(epoch_metrics, dict)
            else getattr(epoch_metrics, "mmd_loss", None)
        )

        if mmd_loss is not None:
            # Ensure prefix is string
            from puffle.Utils.modes import MetricMode

            prefix_str = prefix.value if isinstance(prefix, MetricMode) else prefix
            metrics.setdefault(f"{prefix_str}_mmd_loss", []).append(mmd_loss)

    def _train_one_epoch(self, train_loader, **kwargs):
        """Train one epoch and capture MMD loss."""
        self.running_mmd_loss = 0.0
        # Call parent
        metrics = super()._train_one_epoch(train_loader, **kwargs)

        # Add MMD loss to metrics (convert to dict if it's FairnessMetrics)
        if hasattr(metrics, "to_dict"):
            metrics = metrics.to_dict()

        # Calculate average MMD loss
        num_batches = len(train_loader)
        avg_mmd_loss = self.running_mmd_loss / num_batches if num_batches > 0 else 0.0

        metrics["mmd_loss"] = avg_mmd_loss
        return metrics

    def set_server_predictions(self, y_0: torch.Tensor, y_1: torch.Tensor) -> None:
        """
        Inject tracking sets from the FL server.

        Args:
            y_0: Predictions for demographic group A=0
            y_1: Predictions for demographic group A=1

        """
        self.Y_0 = ensure_tensor(y_0, self.device, dtype=torch.float32)
        self.Y_1 = ensure_tensor(y_1, self.device, dtype=torch.float32)

    def set_client_weights(self, alpha_0: float, alpha_1: float) -> None:
        """
        Set demographic sampling weights.

        These control the proportion of samples contributed to tracking sets,
        not the loss formula.

        Args:
            alpha_0: Weight for A=0 (P_k(A=0) / P(A=0))
            alpha_1: Weight for A=1 (P_k(A=1) / P(A=1))

        """
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1

    def set_total_samples(self, n: int) -> None:
        """
        Set total sample count for N/(N-1) correction.

        Args:
            n: Total number of samples across all clients

        """
        self.N = n

    def set_tracking_function(self, y_0: torch.Tensor, y_1: torch.Tensor) -> None:
        """
        Build the tracking function C(p, A) closure.

        Matches Fair-FL reference implementation's set_C method.

        Args:
            y_0: Tracking set for A=0
            y_1: Tracking set for A=1

        """

        def tracking_fn(p: torch.Tensor, demographic_group: int | None = None):
            """
            Tracking function C(p, A).

            Args:
                p: Predictions (batch_size,) or (batch_size, 1)
                demographic_group: Demographic group (0, 1, or None)

            Returns:
                Scalar tracking value

            """
            if len(p) == 0:
                return 0

            # Compute kernel matrices
            k_0 = distance_kernel(p, y_0).mean(dim=1)
            k_1 = distance_kernel(p, y_1).mean(dim=1)

            if demographic_group is None:
                # No correction
                return (k_0 - k_1).mean()
            if demographic_group == 0:
                # Apply N/(N-1) correction to K(p, Y_0)
                if self.N is not None and self.N > 1:
                    correction = self.N / (self.N - 1)
                    return (k_0 * correction - k_1).mean()
                return (k_0 - k_1).mean()
            if demographic_group == 1:
                # Apply N/(N-1) correction to K(p, Y_1)
                if self.N is not None and self.N > 1:
                    correction = self.N / (self.N - 1)
                    return (k_0 - k_1 * correction).mean()
                return (k_0 - k_1).mean()
            msg = f"Invalid demographic group A={demographic_group}"
            raise ValueError(msg)

        self.tracking_function = tracking_fn

    def _train_batch(
        self,
        batch,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> TrainingBatchResult:
        """
        Train on a single batch with MMD fairness penalty.

        Matches Fair-FL reference: loss = task_loss + 2 * lambda * fairness_loss

        Args:
            batch: (x, z, y) tuple
            model: Neural network
            optimizer: Optimizer
            criterion: Standard task loss (not MixLoss)

        Returns:
            TrainingBatchResult with loss, accuracy, and unfairness metrics

        """
        x_batch = ensure_tensor(batch[0], self.device, dtype=torch.float32)
        z_batch = ensure_tensor(batch[1], self.device)
        y_batch = ensure_tensor(batch[2], self.device)

        optimizer.zero_grad()
        outputs = model(x_batch)

        # 1. Standard task loss
        if outputs.shape[1] > 1:
            # Multi-class
            task_loss = criterion(outputs, y_batch.long())
        else:
            # Binary
            task_loss = criterion(outputs.squeeze(), y_batch.float())

        # Extract predicted probabilities for fairness computation
        if outputs.shape[1] > 1:
            probs = F.softmax(outputs, dim=1)[:, 1]
        else:
            probs = torch.sigmoid(outputs).squeeze()

        # 2. MMD fairness penalty
        fairness_penalty = torch.tensor(0.0, device=self.device)

        if (
            self.lambda_regularization > 0
            and self.Y_0 is not None
            and self.Y_1 is not None
        ):
            mask_0 = z_batch == 0
            mask_1 = z_batch == 1

            probs_0 = probs[mask_0]
            probs_1 = probs[mask_1]

            # Compute C(h_theta(x)) for each group
            c_0 = (
                self.tracking_function(probs_0, demographic_group=0)
                if len(probs_0) > 0
                else 0
            )
            c_1 = (
                self.tracking_function(probs_1, demographic_group=1)
                if len(probs_1) > 0
                else 0
            )

            # Fairness penalty: C(A=0) - C(A=1)
            fairness_penalty = c_0 - c_1

        # 3. Combined loss
        loss = task_loss + 2 * self.lambda_regularization * fairness_penalty

        loss.backward()
        optimizer.step()

        # Accumulate MMD loss for metrics
        if isinstance(fairness_penalty, torch.Tensor):
            self.running_mmd_loss += fairness_penalty.item()
        else:
            self.running_mmd_loss += float(fairness_penalty)

        # 4. Compute metrics
        with torch.no_grad():
            if outputs.shape[1] > 1:
                _, predicted = torch.max(F.softmax(outputs, dim=1), 1)
            else:
                threshold = 0.5
                predicted = (torch.sigmoid(outputs) > threshold).long().squeeze()

            correct_batch = (predicted == y_batch).sum().item()
            total_batch = y_batch.size(0)

            # Compute unfairness for tracking (using PUFFLE's metric)
            # Handle edge case where batch has only one demographic group
            try:
                unfairness_batch, _ = compute_demographic_disparity(
                    z=z_batch,
                    y=predicted,
                    sigma_update_lambda=self.config.sigma_update_lambda,
                    average_probabilities=self.average_probabilities,
                )
            except ValueError:
                # Single demographic group in batch - no disparity to compute
                unfairness_batch = 0.0

        return TrainingBatchResult(
            loss=loss.item(),
            correct=correct_batch,
            total=total_batch,
            y_batch=y_batch,
            predicted=predicted,
            z_batch=z_batch,
            unfairness=unfairness_batch,
        )
