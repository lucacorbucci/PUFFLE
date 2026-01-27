from typing import Any

import dill
import torch
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from Models.utils import get_model
from opacus import PrivacyEngine
from torch import nn
from torch.utils.data import DataLoader

# from Training.training import test, train
from Utils.preferences import Preferences
from Utils.utils import get_optimizer, get_params, set_params

from puffle.PUFFLEModel.puffle_model import PUFFLEModel
from puffle.Regularization.disparity_loss import DisparityRegularizationLoss
from puffle.Regularization.mix_loss import MixLoss


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

        trained_model = get_model(dataset=self.preferences.dataset_name)
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

        model_gc, optimizer_gc, criterion_gc, train_loader_gc = (
            self.privacy_engine.make_private(
                module=trained_model,
                optimizer=optimizer,
                data_loader=self.trainloader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=self.preferences.max_grad_norm,
                criterion=criterion,
                grad_sample_mode="ghost",
                poisson_sampling=True if self.preferences.private_training else False,
            )
        )

        self.model = PUFFLEModel(
            model=model_gc,
            optimizer=optimizer_gc,
            criterion=criterion_gc,
            device=self.device,
            lambda_regularization=self.preferences.regularization_lambda,
            target=self.preferences.target,
            tunable_lambda=self.preferences.regularization_mode == "tunable",
            momentum=self.preferences.momentum
            if self.preferences.momentum is not None
            else None,
            alpha=self.preferences.alpha
            if self.preferences.alpha is not None
            else None,
            weight_decay_alpha=self.preferences.weight_decay_alpha
            if self.preferences.weight_decay_alpha is not None
            else None,
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
        # copy parameters sent by the server into client's local model
        set_params(self.model.model, parameters)

        # do local training (call same function as centralised setting)
        for _ in range(self.preferences.num_epochs):
            result_dict = self.model.train(
                train_loader=self.trainloader, epochs=self.preferences.num_epochs
            )

        # return the model parameters to the server as well as extra info (number of training examples in this case)
        return get_params(self.model.model), len(self.trainloader), result_dict

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
        set_params(self.model.model, parameters)
        result_dict = self.model.evaluate(data_loader=self.valloader)
        return float(result_dict["loss"]), len(self.valloader), {}  # result_dict

    def get_noise_multiplier(self, dataset, target_epsilon=None):
        model_noise = get_model(dataset=self.preferences.dataset_name)
        privacy_engine = PrivacyEngine(accountant="rdp")
        optimizer_noise = nn.CrossEntropyLoss()

        (
            _,
            private_optimizer,
            _,
        ) = privacy_engine.make_private_with_epsilon(
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
