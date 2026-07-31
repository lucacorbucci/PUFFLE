# ABOUTME: Flower client for MMD-Fair FL simulation.
# ABOUTME: Uses MMDFairModel for local training; returns PUFFLE-compatible metric keys.

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._data_loader_fn = data_loader_fn
        self.trainloader = trainloader
        self.valloader = valloader
        self._initialized = False

        self.model: MMDFairModel | None = None
        self.optimizer: torch.optim.Optimizer | None = None

        if self.trainloader is not None and self.valloader is not None:
            self._initialize_model()

    def _initialize_model(self) -> None:
        """
        Perform heavy initialization: model creation, optimizer setup.

        Called lazily on first fit() or eagerly if dataloaders provided.
        """
        if self._initialized:
            return

        if self._data_loader_fn is not None and self.trainloader is None:
            self.trainloader, self.valloader = self._data_loader_fn()

        if self.trainloader is None or self.valloader is None:
            msg = "Data loaders not available"
            raise ValueError(msg)

        trained_model = get_model(
            model_name=self.preferences.model,
            num_classes=self.preferences.num_classes,
            in_channels=self.preferences.in_channels,
        )

        self.optimizer = torch.optim.SGD(
            trained_model.parameters(),
            lr=self.preferences.lr,
            momentum=self.preferences.momentum if self.preferences.momentum else 0.0,
            weight_decay=(
                self.preferences.weight_decay if self.preferences.weight_decay else 0.0
            ),
        )

        use_bce = (self.preferences.num_classes is None) or (
            self.preferences.num_classes == 1
        )

        class LossWrapper(nn.Module):
            def __init__(self, use_bce_loss: bool):
                super().__init__()
                self.use_bce_loss = use_bce_loss
                if self.use_bce_loss:
                    self.criterion = nn.BCEWithLogitsLoss()
                else:
                    self.criterion = nn.CrossEntropyLoss()

            def forward(self, inputs, target):
                # inputs may be a tuple: (outputs, z_batch, lambda_reg)
                if isinstance(inputs, tuple):
                    outputs = inputs[0]
                else:
                    outputs = inputs

                if self.use_bce_loss:
                    return self.criterion(outputs.squeeze(-1), target.float())
                return self.criterion(outputs, target.long())

        criterion = LossWrapper(use_bce_loss=use_bce)

        lambda_fairness = self.preferences.regularization_lambda or 0.0

        # PUFFLEConfig enforces lambda ≤ 1, but MMD-Fair can use larger values.
        # Pass 0 to satisfy the constraint, then set the real value directly.
        config = PUFFLEConfig(lambda_regularization=0.0)

        self.model = MMDFairModel(
            model=trained_model,
            optimizer=self.optimizer,
            criterion=criterion,
            device=self.device,
            config=config,
        )
        self.model.lambda_regularization = lambda_fairness

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
        self._initialize_model()

        if self.model is None or self.optimizer is None:
            msg = "Model not initialized"
            raise ValueError(msg)

        if "Y_0_bytes" in config and "Y_1_bytes" in config:
            Y_0_buffer = io.BytesIO(config["Y_0_bytes"])  # type: ignore
            Y_1_buffer = io.BytesIO(config["Y_1_bytes"])  # type: ignore
            Y_0 = torch.load(Y_0_buffer, weights_only=False)
            Y_1 = torch.load(Y_1_buffer, weights_only=False)

            alpha_0 = float(config.get("alpha_0", 1.0))
            alpha_1 = float(config.get("alpha_1", 1.0))
            N = int(config.get("N", 1))

            self.model.set_server_predictions(Y_0, Y_1)
            self.model.set_client_weights(alpha_0, alpha_1)
            self.model.set_total_samples(N)
            self.model.set_tracking_function(Y_0, Y_1)

        set_params(self.model.model, parameters)
        self.model.model.to(self.device)

        result_dict = self.model.train(
            train_loader=self.trainloader,
            epochs=self.preferences.num_epochs or 1,
        )

        # Learning rate decay matching Fair-FL
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= 0.99

        prediction_samples = self._sample_predictions()

        # Build PUFFLE-compatible metrics (use 0 for unavailable keys)
        metrics: dict[str, Any] = {
            "client_id": self.partition_id,
            "lambda": self.model.lambda_regularization,
        }

        # Add last epoch metrics from model (includes counters, loss, accuracy, etc.)
        for key, value in result_dict.items():
            if isinstance(value, list) and len(value) > 0:
                metrics[key] = (
                    float(value[-1]) if not isinstance(value[-1], dict) else value[-1]
                )
            elif not isinstance(value, (list, dict)):
                metrics[key] = value

        # Add P(A=0) for server alpha weight computation
        if self.trainloader is not None:
            metrics["Pk_A0"] = self._compute_pk_a0()

        if prediction_samples is not None:
            pred_0_buffer = io.BytesIO()
            pred_1_buffer = io.BytesIO()
            torch.save(prediction_samples[0], pred_0_buffer)
            torch.save(prediction_samples[1], pred_1_buffer)
            metrics["pred_0_bytes"] = pred_0_buffer.getvalue()
            metrics["pred_1_bytes"] = pred_1_buffer.getvalue()

        return get_params(self.model.model), len(self.trainloader), metrics

    def _compute_pk_a0(self) -> float:
        """Compute proportion of samples with A=0 in the training set."""
        if self.trainloader is None:
            return 0.5

        total = 0
        count_a0 = 0
        with torch.no_grad():
            for batch in self.trainloader:
                z_batch = batch[1]
                total += len(z_batch)
                count_a0 += (z_batch == 0).sum().item()

        return count_a0 / total if total > 0 else 0.5

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
                tracking_outputs = outputs.squeeze()

                mask_0 = z_batch == 0
                mask_1 = z_batch == 1

                if mask_0.any():
                    preds_0.append(tracking_outputs[mask_0])
                if mask_1.any():
                    preds_1.append(tracking_outputs[mask_1])

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
        Evaluate model on validation/test set.

        Args:
            parameters: Global model parameters
            config: Configuration dict

        Returns:
            Loss, number of examples, PUFFLE-compatible metrics dict

        """
        self._initialize_model()

        if self.model is None or self.valloader is None:
            return 0.0, 0, {}

        set_params(self.model.model, parameters)
        self.model.model.to(self.device)

        self.model.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        y_pred_list = []
        z_list = []

        criterion = self.model.criterion
        device = self.device

        with torch.no_grad():
            for batch in self.valloader:
                x_batch = ensure_tensor(batch[0], device, dtype=torch.float32)
                z_batch = ensure_tensor(batch[1], device)
                y_batch = ensure_tensor(batch[2], device)

                outputs = self.model.model(x_batch)

                if criterion:
                    loss = criterion((outputs, z_batch, 0.0), y_batch)
                    total_loss += loss.item()

                if outputs.shape[1] > 1:
                    _, predicted = torch.max(outputs, 1)
                else:
                    predicted = (torch.sigmoid(outputs).squeeze() > 0.5).long()

                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

                y_pred_list.append(predicted.cpu())
                z_list.append(z_batch.cpu())

        y_pred = torch.cat(y_pred_list).float()
        z = torch.cat(z_list)

        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(self.valloader) if len(self.valloader) > 0 else 0.0

        # Disparity and fairness counters (matches PUFFLE's metric keys)
        mask_z = z == 1
        mask_not_z = z == 0
        counter_z = int(mask_z.sum())
        counter_not_z = int(mask_not_z.sum())
        counter_y_z = int((y_pred[mask_z] == 1).sum()) if counter_z > 0 else 0
        counter_y_not_z = (
            int((y_pred[mask_not_z] == 1).sum()) if counter_not_z > 0 else 0
        )

        p_y_z = counter_y_z / counter_z if counter_z > 0 else 0.0
        p_y_not_z = counter_y_not_z / counter_not_z if counter_not_z > 0 else 0.0
        disparity = abs(p_y_z - p_y_not_z)

        metrics: dict[str, Any] = {
            "client_id": self.partition_id,
            "accuracy": accuracy,
            "loss": avg_loss,
            "disparity": disparity,
            # Counters for aggregated disparity computation
            "counter_z": counter_z,
            "counter_not_z": counter_not_z,
            "counter_y_z": counter_y_z,
            "counter_y_not_z": counter_y_not_z,
            # Dataset counters not tracked in MMD-Fair
            "dataset_counter_z": 0,
            "dataset_counter_not_z": 0,
            "dataset_counter_y_z": 0,
            "dataset_counter_y_not_z": 0,
        }

        return avg_loss, len(self.valloader), metrics

    def get_properties(self, config: dict[str, Scalar]) -> dict[str, Scalar]:
        """
        Return client properties.

        Args:
            config: Configuration dict

        Returns:
            Properties dict with partition_id

        """
        return {"partition_id": self.partition_id}
