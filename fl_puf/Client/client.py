from pathlib import Path

import flwr as fl
import ray
import torch
from fl_puf.Utils.dataset_utils import DatasetDownloader
from fl_puf.Utils.model_utils import Learning
from fl_puf.Utils.utils import Utils
from flwr.common.typing import Scalar
from opacus.validators import ModuleValidator
from torch import nn


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        fed_dir_data: str,
        dataset_name: str,
        DPL: bool,
        DPL_lambda: float,
        private: bool,
        epsilon: float,
        clipping: float,
        epochs: int,
        delta: float,
        lr: float,
    ):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # Instantiate model
        self.lr = lr
        self.net = Utils.get_model(dataset_name)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        self.optimizer_regularization = None
        self.model_regularization = None
        self.private = private
        self.is_private_model_initialized = False
        self.epsilon = epsilon
        self.clipping = clipping

        self.criterion = nn.CrossEntropyLoss()

        self.DPL = DPL
        self.DPL_lambda = DPL_lambda

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_name = dataset_name

    def get_parameters(self, config):
        return Utils.get_params(self.net)

    def fit(self, parameters, config):
        Utils.set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        train_loader = DatasetDownloader.get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=config["batch_size"],
            workers=num_workers,
            dataset=self.dataset_name,
        )
        test_loader = DatasetDownloader.get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=False,
            batch_size=config["batch_size"],
            workers=num_workers,
            dataset=self.dataset_name,
        )

        if self.private and not self.is_private_model_initialized:
            self.net, self.optimizer, strain_loader = Learning.create_private_model(
                model=self.net,
                epsilon=self.epsilon,
                optimizer=self.optimizer,
                train_loader=train_loader,
                epochs=self.epochs,
                delta=self.delta,
                MAX_GRAD_NORM=self.clipping,
            )

            if self.DPL:
                # If we want to use DPL with private training
                # we have to create a second model because
                # we can't just sum the two losses.
                self.model_regularization = Utils.get_model(
                    self.dataset_name,
                    self.device,
                )
                self.model_regularization = ModuleValidator.fix(
                    self.model_regularization,
                )

                self.optimizer_regularization = torch.optim.SGD(
                    self.model_regularization.parameters(),
                    lr=self.lr,
                )

                (
                    self.model_regularization,
                    self.optimizer_regularization,
                    _,
                ) = Learning.create_private_model(
                    model=self.model_regularization,
                    epsilon=self.epsilon,
                    optimizer=self.optimizer_regularization,
                    train_loader=train_loader,
                    epochs=self.epochs,
                    delta=self.delta,
                    MAX_GRAD_NORM=self.gradnorm,
                )
                self.model_regularization.to(self.device)

        # Send model to device
        self.net.to(self.device)

        Learning.train_loop(
            epochs=config["epochs"],
            model=self.net,
            model_regularization=self.model_regularization,
            optimizer=self.optimizer,
            optimizer_regularization=self.optimizer_regularization,
            train_loader=train_loader,
            test_loader=test_loader,
            device=self.device,
            private=self.private,
            criterion=self.criterion,
            DPL=self.DPL,
            DPL_lambda=self.DPL_lambda,
        )

        # Return local model and statistics
        return Utils.get_params(self.net), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        Utils.set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])

        valloader = DatasetDownloader.get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=False,
            batch_size=config["batch_size"],
            workers=num_workers,
            dataset=self.dataset_name,
        )

        # Send model to device
        self.net.to(self.device)

        (
            loss,
            accuracy,
            max_disparity_test,
        ) = Learning.test(
            self.net,
            valloader,
            device=self.device,
            epoch=0,
            DPL_lambda=self.DPL_lambda,
            max_disparity_computation=False,
        )

        # Evaluate
        # loss, accuracy = Utils.test(self.net, valloader, device=self.device)

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}
