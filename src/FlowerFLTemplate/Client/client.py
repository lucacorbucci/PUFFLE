import os
from logging import INFO
from typing import Any

import dill
import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from flwr.common.logger import log
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from torch import nn
from torch.utils.data import DataLoader

from FlowerFLTemplate.Models.utils import get_model

# from Training.training import test, train
from FlowerFLTemplate.Utils.preferences import Preferences
from FlowerFLTemplate.Utils.utils import get_optimizer, get_params, set_params
from puffle.PUFFLEModel.puffle_model import PUFFLEModel
from puffle.Regularization.disparity_loss import DisparityRegularizationLoss
from puffle.Regularization.mix_loss import MixLoss
from puffle.Utils.config import PUFFLEConfig
from puffle.Utils.constants import (
    DEFAULT_ALPHA,
    DEFAULT_MOMENTUM,
    DEFAULT_WEIGHT_DECAY_ALPHA,
)


class FlowerClient(NumPyClient):
    def __init__(
        self,
        trainloader: DataLoader,
        valloader: DataLoader,
        preferences: Preferences,
        partition_id: int,
    ) -> None:
        """
        Initializes a Flower client instance for federated learning.

        Sets up data loaders, device, preferences, and model (SimpleModel for classification or RegressionModel for regression).

        Args:
            trainloader (DataLoader): DataLoader for training data.
            valloader (DataLoader): DataLoader for validation data.
            preferences (Preferences): Configuration preferences for the FL setup.
            partition_id (int): Unique identifier for this client's data partition.

        Returns:
            None

        """
        super().__init__()

        self.partition_id = partition_id
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.preferences = preferences

        self.train_node = False

        with open(f"{self.preferences.fed_dir}/counter_sampling.pkl", "rb") as f:
            counter_sampling = dill.load(f)
            if str(self.partition_id) in counter_sampling:
                self.train_node = True
                self.sampling_frequency = counter_sampling[str(self.partition_id)]

        trained_model = get_model(
            model_name=self.preferences.model,
            num_classes=self.preferences.num_classes,
            in_channels=self.preferences.in_channels,
        )
        optimizer = get_optimizer(trained_model, preferences)
        criterion = MixLoss(
            model_loss=nn.CrossEntropyLoss(),
            unfairness_loss=DisparityRegularizationLoss(),
        )
        self.privacy_engine = PrivacyEngine(accountant="rdp")

        noise_multiplier = (
            self.get_noise_multiplier(
                dataset=self.trainloader, target_epsilon=self.preferences.epsilon
            )
            if self.preferences.private_training
            else 0.0
        )

        model_gc, optimizer_gc, criterion_gc, _train_loader_gc = (
            self.privacy_engine.make_private(
                module=trained_model,
                optimizer=optimizer,
                data_loader=self.trainloader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=self.preferences.max_grad_norm,
                criterion=criterion,
                grad_sample_mode="ghost",
                poisson_sampling=bool(self.preferences.private_training),
            )
        )

        sigma_update_lambda = None
        sigma_statistics = None

        delta = (1 / len(self.trainloader.dataset)) / 2

        if self.preferences.epsilon_lambda is not None:
            sample_rate = self.preferences.batch_size / len(self.trainloader.dataset)
            iterations = self.preferences.num_epochs * len(self.trainloader) * 4
            epsilon_lambda = float(self.preferences.epsilon_lambda)
            sigma_update_lambda = get_noise_multiplier(
                target_epsilon=epsilon_lambda,
                target_delta=delta,
                sample_rate=sample_rate,
                steps=iterations,
                accountant="rdp",
            )

        if self.preferences.epsilon_statistics is not None:
            sample_rate = 1.0
            num_rounds = (
                self.preferences.num_rounds if self.preferences.num_rounds else 1
            )
            iterations = num_rounds * 2
            epsilon_statistics = float(self.preferences.epsilon_statistics)
            sigma_statistics = get_noise_multiplier(
                target_epsilon=epsilon_statistics,
                target_delta=delta,
                sample_rate=sample_rate,
                steps=iterations,
                accountant="rdp",
            )

        self.model = PUFFLEModel(
            model=model_gc,
            optimizer=optimizer_gc,
            criterion=criterion_gc,
            device=self.device,
            config=PUFFLEConfig(
                lambda_regularization=self.preferences.regularization_lambda,
                target=self.preferences.target,
                tunable_lambda=self.preferences.regularization_mode == "tunable",
                momentum=self.preferences.momentum
                if self.preferences.momentum is not None
                else DEFAULT_MOMENTUM,
                alpha=self.preferences.alpha
                if self.preferences.alpha is not None
                else DEFAULT_ALPHA,
                weight_decay_alpha=self.preferences.weight_decay_alpha
                if self.preferences.weight_decay_alpha is not None
                else DEFAULT_WEIGHT_DECAY_ALPHA,
                sigma_update_lambda=sigma_update_lambda,
                sigma_statistics=sigma_statistics,
            ),
        )

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Any]]:
        """
        Performs local training on the client's data using parameters received from the server.

        Updates the local model parameters over the specified number of epochs and returns updated parameters along with training metrics.

        Args:
            parameters (NDArrays): Model parameters from the server.
            config (dict[str, Scalar]): Configuration dictionary from the server.

        Returns:
            tuple[NDArrays, int, dict[str, Any]]: Updated model parameters, number of training examples, and training result dictionary (e.g., containing loss and accuracy).

        Raises:
            RuntimeError: If training fails due to device or model issues.

        """
        # Load average probabilities for DP statistics
        if self.preferences.epsilon_statistics is not None and self.preferences.fed_dir:
            avg_probs_path = os.path.join(self.preferences.fed_dir, "avg_proba.pkl")
            if os.path.exists(avg_probs_path):
                try:
                    with open(avg_probs_path, "rb") as f:
                        avg_probs = dill.load(f)
                    self.model.set_average_probabilities(avg_probs)
                except Exception as e:  # noqa: BLE001
                    log(INFO, f"Failed to load average probabilities: {e}")

        # copy parameters sent by the server into client's local model
        set_params(self.model.model, parameters)

        # do local training (call same function as centralised setting)
        # Note: PUFFLEModel.train() returns dict[str, list[float]] with metrics per epoch
        result_dict = self.model.train(
            train_loader=self.trainloader, epochs=self.preferences.num_epochs
        )

        # Flower expects dict[str, Scalar] where Scalar is bool|bytes|float|int|str
        # PUFFLEModel.train() returns lists (one value per epoch), extract final epoch values
        metrics: dict[str, Any] = {}
        for key, value in result_dict.items():
            if isinstance(value, list) and len(value) > 0:
                # Take the last epoch's value
                metrics[key] = (
                    float(value[-1]) if not isinstance(value[-1], dict) else value[-1]
                )
            elif not isinstance(value, (list, dict)):
                metrics[key] = value

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return get_params(self.model.model), len(self.trainloader), metrics

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Any]]:
        """
        Evaluates the model using parameters received from the server on the client's validation set.

        Computes loss and other metrics (e.g., accuracy for classification, rmse/mae for regression).

        Args:
            parameters (NDArrays): Model parameters from the server.
            config (dict[str, Scalar]): Configuration dictionary from the server.

        Returns:
            tuple[float, int, dict[str, Any]]: Evaluation loss, number of validation examples, and evaluation result dictionary with metrics.

        Raises:
            RuntimeError: If evaluation fails due to device or model issues.

        """
        # Load average probabilities for DP statistics
        if self.preferences.epsilon_statistics is not None and self.preferences.fed_dir:
            avg_probs_path = os.path.join(self.preferences.fed_dir, "avg_proba.pkl")
            if os.path.exists(avg_probs_path):
                try:
                    with open(avg_probs_path, "rb") as f:
                        avg_probs = dill.load(f)
                    self.model.set_average_probabilities(avg_probs)
                except Exception as e:  # noqa: BLE001
                    log(INFO, f"Failed to load average probabilities: {e}")

        set_params(self.model.model, parameters)
        result = self.model.evaluate(data_loader=self.valloader)
        # FairnessMetrics includes loss, accuracy, f1, disparity, and statistics
        # Flower expects dict[str, Scalar] where Scalar is bool|bytes|float|int|str
        # We include the counters from statistics for computing aggregated disparity
        metrics: dict[str, Any] = {
            "loss": result.loss,
            "accuracy": result.accuracy,
            "f1": result.f1,
            "disparity": result.disparity,
        }
        # Add counters for aggregating disparity with statistics
        if result.statistics:
            metrics["counter_z"] = result.statistics.get("counter_z", 0)
            metrics["counter_not_z"] = result.statistics.get("counter_not_z", 0)
            metrics["counter_y_z"] = result.statistics.get("counter_y_z", 0)
            metrics["counter_y_not_z"] = result.statistics.get("counter_y_not_z", 0)
        return float(result.loss), len(self.valloader), metrics

    def get_noise_multiplier(self, dataset, target_epsilon=None):
        model_noise = get_model(
            model_name=self.preferences.model,
            num_classes=self.preferences.num_classes,
            in_channels=self.preferences.in_channels,
        )
        privacy_engine = PrivacyEngine(accountant="rdp")
        optimizer_noise = torch.optim.SGD(model_noise.parameters(), lr=0.1)

        (
            _,
            private_optimizer,
            _,
        ) = privacy_engine.make_private_with_epsilon(  # type: ignore
            module=model_noise,
            optimizer=optimizer_noise,
            data_loader=dataset,
            epochs=self.sampling_frequency * self.train_parameters.epochs,
            target_epsilon=self.train_parameters.epsilon
            if target_epsilon is None
            else target_epsilon,
            target_delta=self.delta,
            max_grad_norm=self.clipping,
        )

        return private_optimizer.noise_multiplier
