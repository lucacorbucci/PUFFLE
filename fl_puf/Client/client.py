import copy
import gc
import os
import random
import warnings
from pathlib import Path

import dill
import flwr as fl
import numpy as np
import ray
import torch
from DPL.learning import Learning
from DPL.Utils.model_utils import ModelUtils
from DPL.Utils.train_parameters import TrainParameters
from fl_puf.Utils.utils import Utils
from flwr.common.typing import Scalar


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        fed_dir_data: str,
        dataset_name: str,
        clipping: float,
        delta: float,
        lr: float,
        train_parameters: TrainParameters,
    ):
        print(f"Node {cid} is initializing...")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.train_parameters = copy.deepcopy(train_parameters)
        # self.train_parameters.wandb_run = None
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.dataset_name = dataset_name
        self.clipping = clipping
        self.delta = delta
        self.lr = lr

        if (
            os.path.exists(f"{self.fed_dir}/random_state_random_{self.cid}.pkl")
            and os.path.exists(f"{self.fed_dir}/random_state_np_{self.cid}.pkl")
            and os.path.exists(f"{self.fed_dir}/random_state_torch_{self.cid}.pkl")
            and os.path.exists(f"{self.fed_dir}/random_state_torch_cuda_{self.cid}.pkl")
        ):
            print(f"Loading State from disk {self.cid}")
            with open(
                f"{self.fed_dir}/random_state_random_{self.cid}.pkl", "rb"
            ) as file:
                state = dill.load(file)
                random.setstate(state)

            with open(f"{self.fed_dir}/random_state_np_{self.cid}.pkl", "rb") as file:
                state = dill.load(file)
                np.random.set_state(state)

            with open(
                f"{self.fed_dir}/random_state_torch_{self.cid}.pkl", "rb"
            ) as file:
                state = dill.load(file)
                torch.set_rng_state(state)

            with open(
                f"{self.fed_dir}/random_state_torch_cuda_{self.cid}.pkl", "rb"
            ) as file:
                state = dill.load(file)
                torch.cuda.set_rng_state(state)
        else:
            print(f"Seeding Node {cid}")

            torch.manual_seed(self.train_parameters.seed)
            random.seed(self.train_parameters.seed)
            np.random.seed(self.train_parameters.seed)
            torch.cuda.manual_seed(self.train_parameters.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.net = ModelUtils.get_model(
            dataset_name, device=self.train_parameters.device
        )
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.lr,
        )

        if self.train_parameters.DPL:  # and self.train_parameters.private:
            self.model_regularization = ModelUtils.get_model(
                self.dataset_name,
                device=self.train_parameters.device,
            )
            self.optimizer_regularization = torch.optim.SGD(
                self.model_regularization.parameters(),
                lr=self.lr,
            )

        print(f"Epsilon: {self.train_parameters.epsilon}, Clipping: {clipping}")

    def get_parameters(self, config):
        return Utils.get_params(self.net)

    def write_state(state, path):
        with open(path, "wb") as f:
            dill.dump(state, f)


    def fit(self, parameters, config):
        Utils.set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        train_loader = Utils.get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=config["batch_size"],
            workers=num_workers,
            dataset=self.dataset_name,
        )

        (
            private_net,
            private_optimizer,
            train_loader,
        ) = Utils.create_private_model(
            model=self.net,
            epsilon=self.train_parameters.epsilon,
            original_optimizer=self.optimizer,
            train_loader=train_loader,
            epochs=self.train_parameters.epochs,
            delta=self.delta,
            MAX_GRAD_NORM=self.clipping,
            batch_size=self.train_parameters.batch_size,
        )
        private_net.to(self.train_parameters.device)

        private_model_regularization = None
        private_optimizer_regularization = None

        if self.train_parameters.DPL:  # and self.train_parameters.private:
            (
                private_model_regularization,
                private_optimizer_regularization,
                _,
            ) = Utils.create_private_model(
                model=self.model_regularization,
                epsilon=self.train_parameters.epsilon,
                original_optimizer=self.optimizer_regularization,
                train_loader=train_loader,
                epochs=self.train_parameters.epochs,
                delta=self.delta,
                MAX_GRAD_NORM=self.clipping,
                batch_size=self.train_parameters.batch_size,
            )
            private_model_regularization.to(self.train_parameters.device)
            print(f"Created private model for regularization on node {self.cid}")

        gc.collect()

        for epoch in range(0, self.train_parameters.epochs):
            Learning.train_private_model(
                train_parameters=self.train_parameters,
                model=private_net,
                model_regularization=private_model_regularization,
                optimizer=private_optimizer,
                optimizer_regularization=private_optimizer_regularization,
                train_loader=train_loader,
                test_loader=None,
                current_epoch=epoch,
            )

        Utils.set_params(self.net, Utils.get_params(private_net))

        del private_net
        if private_model_regularization:
            del private_model_regularization
        gc.collect()

        # store the random_state on disk
        with open(f"{self.fed_dir}/random_state_random_{self.cid}.pkl", "wb") as f:
            dill.dump(random.getstate(), f)
        with open(f"{self.fed_dir}/random_state_np_{self.cid}.pkl", "wb") as f:
            dill.dump(np.random.get_state(), f)
        with open(f"{self.fed_dir}/random_state_torch_{self.cid}.pkl", "wb") as f:
            dill.dump(torch.get_rng_state(), f)
        with open(f"{self.fed_dir}/random_state_torch_cuda_{self.cid}.pkl", "wb") as f:
            dill.dump(torch.cuda.get_rng_state(device=self.train_parameters.device), f)

        # Return local model and statistics
        return Utils.get_params(self.net), len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        Utils.set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        valloader = Utils.get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=False,
            batch_size=self.train_parameters.batch_size,
            workers=num_workers,
            dataset=self.dataset_name,
        )

        # Send model to device
        self.net.to(self.train_parameters.device)

        # Evaluate
        (
            test_loss,
            accuracy,
            f1score,
            precision,
            recall,
            max_disparity_test,
        ) = Learning.test(
            model=self.net,
            test_loader=valloader,
            train_parameters=self.train_parameters,
            current_epoch=None,
        )

        self.net.to("cpu")
        gc.collect()

        # Return statistics
        return float(test_loss), len(valloader.dataset), {"accuracy": float(accuracy)}
