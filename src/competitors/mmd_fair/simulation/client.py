# ABOUTME: Flower client for MMD-Fair FL simulation.
# ABOUTME: Uses MMDFairModel instead of PUFFLEModel, no privacy engine, BCEWithLogitsLoss.

import io
from collections.abc import Callable
from typing import Any

import torch
from FlowerFLTemplate.Models.utils import get_model
from FlowerFLTemplate.Utils.preferences import Preferences
from FlowerFLTemplate.Utils.utils import get_params, set_params
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from puffle.Utils.config import PUFFLEConfig
from puffle.Utils.tensor_utils import ensure_tensor
from torch import nn
from torch.utils.data import DataLoader

from competitors.mmd_fair.model import MMDFairModel


class MMDFairFlowerClient(NumPyClient):
    """
    Flower client for MMD-Fair FL simulation.

    Uses lazy initialization: data loading and model creation are deferred
    until the first fit() call.
    """

    def __init__(
        self,
        partition_id: int,
        preferences: Preferences,
        data_loader_fn: Callable[[], tuple[DataLoader, DataLoader]] | None = None,
        trainloader: DataLoader | None = None,
        valloader: DataLoader | None = None,
    ):
        """
        Initialize MMD-Fair Flower client.

        Args:
            partition_id: Client partition ID
            preferences: FL configuration
            data_loader_fn: Lazy data loading function
            trainloader: Training data loader (eager mode)
            valloader: Validation data loader (eager mode)

        """
        self.partition_id = partition_id
        self.preferences = preferences
        self.device = (
            "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Lazy loading support
        self._data_loader_fn = data_loader_fn
        self.trainloader = trainloader
        self.valloader = valloader
        self._initialized = False

        # Model components (created in _initialize_model)
        self.model: MMDFairModel | None = None
        self.optimizer: torch.optim.Optimizer | None = None

        # If dataloaders provided eagerly, initialize now
        if self.trainloader is not None and self.valloader is not None:
            self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Perform heavy initialization: model creation, optimizer setup.

        Called lazily on first fit() or eagerly if dataloaders provided.
        """
        if self._initialized:
            return

        # Load data if using lazy loading
        if self._data_loader_fn is not None and self.trainloader is None:
            self.trainloader, self.valloader = self._data_loader_fn()

        if self.trainloader is None or self.valloader is None:
            msg = "Data loaders not available"
            raise ValueError(msg)

        # Create model
        trained_model = get_model(
            model_name=self.preferences.model,
            num_classes=self.preferences.num_classes,
            in_channels=self.preferences.in_channels,
        )

        # Plain SGD optimizer (no privacy engine)
        self.optimizer = torch.optim.SGD(
            trained_model.parameters(),
            lr=self.preferences.lr,
            momentum=self.preferences.momentum if self.preferences.momentum else 0.0,
            weight_decay=self.preferences.weight_decay
            if self.preferences.weight_decay
            else 0.0,
        )

        # Select loss based on model output dimension (1=BCE, >1=CE)
        use_bce = (self.preferences.num_classes is None) or (
            self.preferences.num_classes == 1
        )

        # Wrap loss to handle tuple input from PUFFLEModel.evaluate
        # Parent class passes (outputs, z_batch, lambda) but standard loss only needs (outputs, target)
        class LossWrapper(nn.Module):
            def __init__(self, use_bce_loss: bool):
                super().__init__()
                self.use_bce_loss = use_bce_loss
                if self.use_bce_loss:
                    self.criterion = nn.BCEWithLogitsLoss()
                else:
                    self.criterion = nn.CrossEntropyLoss()

            def forward(self, inputs, target):
                # inputs is a tuple: (outputs, z_batch, lambda_reg)
                if isinstance(inputs, tuple):
                    outputs = inputs[0]
                else:
                    outputs = inputs

                if self.use_bce_loss:
                    return self.criterion(outputs, target.float())
                return self.criterion(outputs, target.long())

        criterion = LossWrapper(use_bce_loss=use_bce)

        # Create MMDFairModel
        config = PUFFLEConfig(
            lambda_regularization=self.preferences.regularization_lambda,
        )

        self.model = MMDFairModel(
            model=trained_model,
            optimizer=self.optimizer,
            criterion=criterion,
            device=self.device,
            config=config,
        )

        self._initialized = True

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Any]]:
        """
        Perform local training.

        Args:
            parameters: Global model parameters from server
            config: Configuration dict with Y_0/Y_1 tracking sets

        Returns:
            Updated parameters, number of examples, metrics dict

        """
        # Lazy initialization
        self._initialize_model()

        if self.model is None or self.optimizer is None:
            msg = "Model not initialized"
            raise ValueError(msg)

        # Deserialize Y_0, Y_1, alpha weights, N from config
        if "Y_0_bytes" in config and "Y_1_bytes" in config:
            Y_0_buffer = io.BytesIO(config["Y_0_bytes"])  # type: ignore
            Y_1_buffer = io.BytesIO(config["Y_1_bytes"])  # type: ignore
            Y_0 = torch.load(Y_0_buffer, weights_only=False)
            Y_1 = torch.load(Y_1_buffer, weights_only=False)

            alpha_0 = float(config.get("alpha_0", 1.0))
            alpha_1 = float(config.get("alpha_1", 1.0))
            N = int(config.get("N", 1))

            # Set server state on model
            self.model.set_server_predictions(Y_0, Y_1)
            self.model.set_client_weights(alpha_0, alpha_1)
            self.model.set_total_samples(N)
            self.model.set_tracking_function(Y_0, Y_1)

        # Load global parameters
        set_params(self.model.model, parameters)

        # Local training
        result_dict = self.model.train(
            train_loader=self.trainloader,
            epochs=self.preferences.num_epochs or 1,
        )

        # Apply learning rate decay (0.99× per round, matching Fair-FL)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= 0.99

        # Sample predictions for server's update_C
        prediction_samples = self._sample_predictions()

        # Prepare metrics
        metrics: dict[str, Any] = {
            "client_id": self.partition_id,
            "lambda": self.model.lambda_regularization,
        }

        # Add last epoch metrics
        for key, value in result_dict.items():
            if isinstance(value, list) and len(value) > 0:
                metrics[key] = (
                    float(value[-1]) if not isinstance(value[-1], dict) else value[-1]
                )
            elif not isinstance(value, (list, dict)):
                metrics[key] = value

        # Serialize prediction samples as bytes
        if prediction_samples is not None:
            pred_0_buffer = io.BytesIO()
            pred_1_buffer = io.BytesIO()
            torch.save(prediction_samples[0], pred_0_buffer)
            torch.save(prediction_samples[1], pred_1_buffer)
            metrics["pred_0_bytes"] = pred_0_buffer.getvalue()
            metrics["pred_1_bytes"] = pred_1_buffer.getvalue()

        return get_params(self.model.model), len(self.trainloader), metrics

    def _sample_predictions(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Sample predictions from local data for server's update_C.

        Returns:
            Tuple of (predictions_A0, predictions_A1) or None

        """
        if self.model is None or self.trainloader is None:
            return None

        self.model.model.eval()
        preds_0 = []
        preds_1 = []

        with torch.no_grad():
            for batch in self.trainloader:
                x_batch = batch[0].to(self.device)
                z_batch = batch[1].to(self.device)

                outputs = self.model.model(x_batch)
                probs = torch.sigmoid(outputs).squeeze()

                # Split by demographic group
                mask_0 = z_batch == 0
                mask_1 = z_batch == 1

                if mask_0.any():
                    preds_0.append(probs[mask_0])
                if mask_1.any():
                    preds_1.append(probs[mask_1])

        pred_0_tensor = (
            torch.cat(preds_0) if preds_0 else torch.tensor([], device=self.device)
        )
        pred_1_tensor = (
            torch.cat(preds_1) if preds_1 else torch.tensor([], device=self.device)
        )

        return pred_0_tensor, pred_1_tensor

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Any]]:
        """
        Evaluate model on validation set.

        Args:
            parameters: Global model parameters
            config: Configuration dict

        Returns:
            Loss, number of examples, metrics dict

        """
        self._initialize_model()

        if self.model is None or self.valloader is None:
            return 0.0, 0, {}

        set_params(self.model.model, parameters)

        # Manual evaluation to compute Fair-FL P1 and PUFFLE unfairness
        self.model.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        y_true_list = []
        y_pred_probs_list = []
        z_list = []

        criterion = self.model.criterion
        device = self.device

        with torch.no_grad():
            for batch in self.valloader:
                x_batch = ensure_tensor(batch[0], device, dtype=torch.float32)
                z_batch = ensure_tensor(batch[1], device)
                y_batch = ensure_tensor(batch[2], device)

                outputs = self.model.model(x_batch)

                # Handle loss wrapper tuple requirement
                if criterion:
                    # criterion expects (outputs, z, lambda) tuple
                    loss = criterion((outputs, z_batch, 0.0), y_batch)
                    total_loss += loss.item()

                if outputs.shape[1] > 1:
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    _, predicted = torch.max(outputs, 1)
                else:
                    probs = torch.sigmoid(outputs).squeeze()
                    predicted = (probs > 0.5).long()

                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

                y_true_list.append(y_batch.cpu())
                y_pred_probs_list.append(probs.cpu())
                z_list.append(z_batch.cpu())

        y_probs = torch.cat(y_pred_probs_list)
        z = torch.cat(z_list)
        y_pred = (y_probs > 0.5).float()

        # Compute Metrics
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.valloader) if len(self.valloader) > 0 else 0.0

        # 1. PUFFLE Unfairness (Demographic Disparity)
        from puffle.Utils.metric import compute_demographic_disparity

        unfairness, _ = compute_demographic_disparity(z=z, y=y_pred, is_validation=True)

        # 2. Fair-FL P1 Metric
        p1_metric = compute_fair_fl_p1(y_probs, z)

        # 3. MMD Loss Term (Fairness Penalty)
        # We need to compute MMD on the validation set
        # Using the model's tracking function if available, or just raw MMD on the batch?
        # Fair-FL logs MMD(Y_hat[A==0], Y_hat[A==1]) on the test set.
        # We can implement a simple MMD helper here.
        mmd_loss = compute_mmd_loss(y_probs, z)

        metrics = {
            "client_id": self.partition_id,
            "accuracy": accuracy,
            "loss": avg_loss,
            "unfairness": float(unfairness),
            "fair_fl_p1": float(p1_metric),
            "mmd_loss": float(mmd_loss),
            "val_loss": avg_loss,
            "val_acc": accuracy,
            "val_fairness": float(unfairness),
            "val_fair_fl_p1": float(p1_metric),
            "val_mmd_loss": float(mmd_loss),
        }

        return avg_loss, len(self.valloader), metrics

    def get_properties(self, config: dict[str, Scalar]) -> dict[str, Scalar]:
        """
        Return client properties.

        Lightweight method that doesn't trigger initialization.

        Args:
            config: Configuration dict

        Returns:
            Properties dict with partition_id

        """
        return {"partition_id": self.partition_id}


def compute_fair_fl_p1(probs: torch.Tensor, z: torch.Tensor) -> float:
    """
    Compute P1 metric from Fair-FL.
    |P(Y=1|A=0) - P(Y=1|A=1)|
    """
    if len(probs) == 0:
        return 0.0

    y_pred = (probs > 0.5).float()

    mask_0 = z == 0
    mask_1 = z == 1

    p_y1_z0 = y_pred[mask_0].mean() if mask_0.any() else torch.tensor(0.0)
    p_y1_z1 = y_pred[mask_1].mean() if mask_1.any() else torch.tensor(0.0)

    return abs(float(p_y1_z0 - p_y1_z1))


def compute_mmd_loss(probs: torch.Tensor, z: torch.Tensor) -> float:
    """
    Compute MMD loss on the validation set.
    """
    mask_0 = z == 0
    mask_1 = z == 1

    if not mask_0.any() or not mask_1.any():
        return 0.0  # Cannot compute MMD with only one group

    p0 = probs[mask_0]
    p1 = probs[mask_1]

    from competitors.mmd_fair.model import distance_kernel

    # MMD = E[K(X,X)] + E[K(Y,Y)] - 2E[K(X,Y)]
    # Use simpler estimation for validation logging
    k_00 = distance_kernel(p0, p0).mean()
    k_11 = distance_kernel(p1, p1).mean()
    k_01 = distance_kernel(p0, p1).mean()

    return float(k_00 + k_11 - 2 * k_01)
