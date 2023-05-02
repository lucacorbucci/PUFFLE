from pathlib import Path

import flwr as fl
import ray
import torch
from dataset_utils import DatasetDownloader
from flwr.common.typing import Scalar
from model_utils import Learning
from torch import nn
from utils import Utils


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str, dataset_name: str):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}

        # Instantiate model
        self.net = Utils.get_model(dataset_name)
        self.model_regularization = None
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01)
        self.optimizer_regularization = None

        self.private = False
        self.criterion = nn.CrossEntropyLoss()

        self.DPL = False
        self.DPL_lambda = None
        self.loss_lambda = None

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dataset_name = dataset_name

    def get_parameters(self, config):
        return Utils.get_params(self.net)

    def fit(self, parameters, config):
        Utils.set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        trainloader = DatasetDownloader.get_dataloader(
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

        # Send model to device
        self.net.to(self.device)

        # Train
        # Utils.train(self.net, trainloader, epochs=config["epochs"], device=self.device)

        # # if config["DPL"]:
        Learning.train_loop(
            epochs=config["epochs"],
            model=self.net,
            model_regularization=self.model_regularization,
            optimizer=self.optimizer,
            optimizer_regularization=self.optimizer_regularization,
            train_loader=trainloader,
            test_loader=test_loader,
            device=self.device,
            private=self.private,
            criterion=self.criterion,
            DPL=self.DPL,
            wandb_run=None,
            DPL_lambda=self.DPL_lambda,
            loss_lambda=self.loss_lambda,
        )
        # else:
        #     # Train
        #     ModelUtils.train(
        #         self.net,
        #         trainloader,
        #         epochs=config["epochs"],
        #         device=self.device,
        #     )

        # Return local model and statistics
        return Utils.get_params(self.net), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        Utils.set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])

        valloader = DatasetDownloader.get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=False,
            batch_size=50,
            workers=num_workers,
            dataset=self.dataset_name,
        )

        # Send model to device
        self.net.to(self.device)

        # Evaluate
        loss, accuracy = Utils.test(self.net, valloader, device=self.device)

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}
