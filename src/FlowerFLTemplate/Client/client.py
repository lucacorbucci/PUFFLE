import os
from collections.abc import Callable
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
from FlowerFLTemplate.Utils.preferences import Preferences
from FlowerFLTemplate.Utils.utils import get_optimizer, get_params, set_params
from puffle.PUFFLEModel.puffle_model import PUFFLEModel
from puffle.Regularization.disparity_loss import DisparityRegularizationLoss
from puffle.Regularization.error_rate_regularization_loss import (
    ErrorRateRegularizationLoss,
)
from puffle.Regularization.mix_loss import MixLoss
from puffle.Utils.config import PUFFLEConfig
from puffle.Utils.constants import (
    DEFAULT_ALPHA,
    DEFAULT_MOMENTUM,
    DEFAULT_WEIGHT_DECAY_ALPHA,
)


class FlowerClient(NumPyClient):
    """
    Flower client with lazy initialization.

    This client defers heavy operations (data loading, model creation) until the first
    fit() or evaluate() call. This allows get_properties() to respond instantly during
    registration, enabling proper partition_id-to-cid mapping.
    """

    def __init__(
        self,
        partition_id: int,
        preferences: Preferences,
        data_loader_fn: Callable[[], tuple[DataLoader, DataLoader]] | None = None,
        trainloader: DataLoader | None = None,
        valloader: DataLoader | None = None,
    ) -> None:
        """
        Initializes a Flower client instance for federated learning.

        If data_loader_fn is provided, data loading is deferred (lazy mode).
        If trainloader/valloader are provided directly, operates in eager mode.

        Args:
            partition_id (int): Unique identifier for this client's data partition.
            preferences (Preferences): Configuration preferences for the FL setup.
            data_loader_fn (Callable): Function that returns (trainloader, valloader).
                                       If provided, enables lazy loading.
            trainloader (DataLoader): DataLoader for training data (eager mode).
            valloader (DataLoader): DataLoader for validation data (eager mode).

        Returns:
            None

        """
        super().__init__()

        self.partition_id = partition_id
        self.preferences = preferences
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Lazy loading support
        self._data_loader_fn = data_loader_fn
        self._initialized = False

        # These will be set during initialization
        self.trainloader = trainloader
        self.valloader = valloader
        self.model: PUFFLEModel | None = None
        self.train_node = False
        self.sampling_frequency: int | None = None
        self.privacy_engine: PrivacyEngine | None = None

        # If dataloaders were provided directly, we're in eager mode
        if trainloader is not None and valloader is not None:
            self._initialize_model()

    def _initialize_model(self, phase: str) -> None:
        """
        Performs the heavy initialization: model creation, privacy engine setup, etc.
        Called lazily on first fit()/evaluate() or eagerly if dataloaders provided.
        """
        if self._initialized:
            return

        # If using lazy loading, load data now
        if self._data_loader_fn is not None and self.trainloader is None:
            self.trainloader, self.valloader = self._data_loader_fn()

        if self.trainloader is None or self.valloader is None:
            msg = "Data loaders not available. Provide data_loader_fn or trainloader/valloader."
            raise ValueError(msg)

        # Load counter sampling info
        counter_sampling_path = f"{self.preferences.fed_dir}/counter_sampling.pkl"
        if os.path.exists(counter_sampling_path):
            with open(counter_sampling_path, "rb") as f:
                counter_sampling = dill.load(f)
                if str(self.partition_id) in counter_sampling:
                    self.train_node = True
                    self.sampling_frequency = counter_sampling[str(self.partition_id)]

        # Create and wrap model with privacy engine
        trained_model = get_model(
            model_name=self.preferences.model,
            num_classes=self.preferences.num_classes,
            in_channels=self.preferences.in_channels,
        )
        optimizer = get_optimizer(trained_model, self.preferences)

        # Select fairness loss based on configuration
        if self.preferences.fairness_metric == "error_rate":
            unfairness_loss = ErrorRateRegularizationLoss()
        else:
            unfairness_loss = DisparityRegularizationLoss()

        criterion = MixLoss(
            model_loss=nn.CrossEntropyLoss(),
            unfairness_loss=unfairness_loss,
        )
        self.privacy_engine = PrivacyEngine(accountant="rdp")

        if os.path.exists(
            f"{self.preferences.fed_dir}/accountant_{self.partition_id}.pkl"
        ):
            with open(
                f"{self.preferences.fed_dir}/accountant_{self.partition_id}.pkl", "rb"
            ) as file:
                accountant = dill.load(file)
                self.privacy_engine.accountant = accountant

        noise_multiplier = (
            self._get_noise_multiplier_internal(
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

        if self.preferences.epsilon_lambda is not None and phase == "train":
            sampling_ratio = 1 / len(self.trainloader)

            # TODO: we should probably add + self.sampling_frequency*4
            # to support the first update of the lambda
            iterations = (
                self.sampling_frequency
                * self.preferences.num_epochs
                * len(self.trainloader)
                * 4
            ) + self.sampling_frequency * 4
            sigma_update_lambda = get_noise_multiplier(
                target_epsilon=self.preferences.epsilon_lambda,
                target_delta=delta,
                sample_rate=sampling_ratio,
                steps=iterations,
                accountant="rdp",
            )

        if self.preferences.epsilon_statistics is not None and phase == "train":
            sampling_ratio = 1
            # we multiply by 2 because every time we send two values
            iterations = self.sampling_frequency * 2 * 2
            epsilon_statistics = float(self.preferences.epsilon_statistics)
            sigma_statistics = get_noise_multiplier(
                target_epsilon=epsilon_statistics,
                target_delta=delta,
                sample_rate=sampling_ratio,
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

        self._initialized = True
        log(INFO, f"Client {self.partition_id} initialized (lazy)")

    def _get_noise_multiplier_internal(
        self, dataset: DataLoader, target_epsilon: float | None = None
    ) -> float:
        """
        Calculate the noise multiplier for a given target epsilon.
        Internal version used during initialization.
        """
        if not self.train_node or self.sampling_frequency is None:
            return 0.0

        model_noise = get_model(
            model_name=self.preferences.model,
            num_classes=self.preferences.num_classes,
            in_channels=self.preferences.in_channels,
        )
        privacy_engine = PrivacyEngine(accountant="rdp")
        optimizer_noise = torch.optim.SGD(model_noise.parameters(), lr=0.1)

        delta = (1 / len(dataset.dataset)) / 2  # type: ignore[arg-type]

        (
            _,
            private_optimizer,
            _,
        ) = privacy_engine.make_private_with_epsilon(  # type: ignore[misc]
            module=model_noise,
            optimizer=optimizer_noise,
            data_loader=dataset,
            epochs=self.sampling_frequency * self.preferences.num_epochs,
            target_epsilon=self.preferences.epsilon
            if target_epsilon is None
            else target_epsilon,
            target_delta=delta,
            max_grad_norm=self.preferences.max_grad_norm,
        )

        return private_optimizer.noise_multiplier

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Any]]:
        """
        Performs local training on the client's data using parameters received from the server.

        Updates the local model parameters over the specified number of epochs and returns
        updated parameters along with training metrics.

        Args:
            parameters (NDArrays): Model parameters from the server.
            config (dict[str, Scalar]): Configuration dictionary from the server.

        Returns:
            tuple[NDArrays, int, dict[str, Any]]: Updated model parameters, number of
                training examples, and training result dictionary.

        """
        # Lazy initialization on first fit() call
        self._initialize_model(phase="train")

        avg_probs = None
        # Load average probabilities for DP statistics
        if self.preferences.fed_dir:
            avg_probs_path = os.path.join(self.preferences.fed_dir, "avg_proba.pkl")
            if os.path.exists(avg_probs_path):
                try:
                    with open(avg_probs_path, "rb") as f:
                        avg_probs = dill.load(f)
                    self.model.set_average_probabilities(avg_probs)
                except (OSError, ValueError, KeyError) as e:
                    log(INFO, f"Failed to load average probabilities: {e}")
            else:
                avg_probs = {
                    "first_round": True,
                }
                self.model.set_average_probabilities(avg_probs)

        # copy parameters sent by the server into client's local model
        set_params(self.model.model, parameters)

        # Initialize lambda from inference (skipped if avg_probs is None or has first_round flag)
        if self.model.tunable_lambda:
            # Determine if this is first round
            is_first_round = avg_probs is None or avg_probs.get("first_round", False)
            avg_probs_for_init = None if is_first_round else avg_probs

            initial_lambda = self.model.initialize_lambda_from_inference(
                data_loader=self.trainloader,
                average_probabilities=avg_probs_for_init,
                sigma_update_lambda=self.model.config.sigma_update_lambda,
            )
            log(
                INFO,
                f"Client {self.partition_id} initialized lambda={initial_lambda:.4f} "
                f"(first_round={is_first_round})",
            )

        # do local training (call same function as centralised setting)
        result_dict = self.model.train(
            train_loader=self.trainloader,
            epochs=self.preferences.num_epochs,
            average_probabilities=avg_probs,
        )

        # Flower expects dict[str, Scalar] where Scalar is bool|bytes|float|int|str
        metrics: dict[str, Any] = {}
        for key, value in result_dict.items():
            if isinstance(value, list) and len(value) > 0:
                # Take the last epoch's value
                metrics[key] = (
                    float(value[-1]) if not isinstance(value[-1], dict) else value[-1]
                )
            elif not isinstance(value, (list, dict)):
                metrics[key] = value

        metrics["client_id"] = self.partition_id
        metrics["lambda"] = self.model.lambda_regularization

        # We need to store the state of the privacy engine and all the
        # details about the private training
        with open(
            f"{self.preferences.fed_dir}/accountant_{self.partition_id}.pkl", "wb"
        ) as f:
            dill.dump(self.privacy_engine.accountant, f)

        return get_params(self.model.model), len(self.trainloader), metrics

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Any]]:
        """
        Evaluates the model using parameters received from the server on the client's validation set.

        Args:
            parameters (NDArrays): Model parameters from the server.
            config (dict[str, Scalar]): Configuration dictionary from the server.

        Returns:
            tuple[float, int, dict[str, Any]]: Evaluation loss, number of validation
                examples, and evaluation result dictionary with metrics.

        """
        # Lazy initialization on first evaluate() call
        self._initialize_model(phase="evaluate")

        set_params(self.model.model, parameters)
        result = self.model.evaluate(data_loader=self.valloader, _is_validation=True)

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
            metrics["counter_not_y_z"] = result.statistics.get("counter_not_y_z", 0)
            metrics["counter_not_y_not_z"] = result.statistics.get(
                "counter_not_y_not_z", 0
            )
            metrics["counter_y"] = result.statistics.get("counter_y", 0)
            metrics["counter_not_y"] = result.statistics.get("counter_not_y", 0)
            metrics["total_samples"] = result.statistics.get("total_samples", 0)

        # Add dataset statistics (Ground Truth)
        if result.dataset_statistics:
            ds = result.dataset_statistics
            metrics["dataset_counter_z"] = ds.get("counter_z", 0)
            metrics["dataset_counter_not_z"] = ds.get("counter_not_z", 0)
            metrics["dataset_counter_y_z"] = ds.get("counter_y_z", 0)
            metrics["dataset_counter_y_not_z"] = ds.get("counter_y_not_z", 0)
            metrics["dataset_counter_y"] = ds.get("counter_y", 0)
            metrics["dataset_counter_not_y"] = ds.get("counter_not_y", 0)

        metrics["client_id"] = self.partition_id

        return float(result.loss), len(self.valloader), metrics

    def get_noise_multiplier(
        self, dataset: DataLoader, target_epsilon: float | None = None
    ) -> float:
        """
        Calculate the noise multiplier for a given target epsilon.

        Args:
            dataset (Dataset): The dataset used for training.
            target_epsilon (float, optional): The target epsilon for differential privacy.

        Returns:
            float: The calculated noise multiplier.

        """
        return self._get_noise_multiplier_internal(dataset, target_epsilon)

    def get_properties(self, config: dict[str, Scalar]) -> dict[str, Scalar]:
        """
        Return client's properties. This is called during registration.

        This method is intentionally lightweight and does NOT trigger initialization.
        It only returns the partition_id, allowing fast registration.

        Args:
            config: Configuration parameters requested by the server.

        Returns:
            dict: Properties including partition_id.

        """
        log(INFO, f"Client {self.partition_id} get_properties called")
        return {"partition_id": self.partition_id}
