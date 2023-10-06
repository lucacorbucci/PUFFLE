import copy
import gc
import logging
import os
import random
import warnings
from collections import Counter
from pathlib import Path

import dill
import flwr as fl
import numpy as np
import ray
import torch
from DPL.learning import Learning
from DPL.RegularizationLoss import RegularizationLoss
from DPL.Utils.model_utils import ModelUtils
from fl_puf.Utils.utils import Utils
from flwr.common.typing import Scalar
from Utils.train_parameters import TrainParameters


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
        logging.info(f"Node {cid} is initializing...")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.train_parameters = copy.deepcopy(train_parameters)
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.dataset_name = dataset_name
        self.clipping = clipping
        self.delta = delta
        self.lr = lr

        self.net = ModelUtils.get_model(
            dataset_name, device=self.train_parameters.device
        )
        self.optimizer = self.get_optimizer(model=self.net)

        if self.train_parameters.DPL:
            self.model_regularization = ModelUtils.get_model(
                self.dataset_name,
                device=self.train_parameters.device,
            )
            self.optimizer_regularization = self.get_optimizer(
                model=self.model_regularization
            )

    def get_optimizer(self, model):
        if self.train_parameters.optimizer == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=self.lr,
            )
        elif self.train_parameters.optimizer == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=self.lr,
            )
        elif self.train_parameters.optimizer == "adamW":
            return torch.optim.AdamW(
                model.parameters(),
                lr=self.lr,
            )
        else:
            raise ValueError("Optimizer not recognized")

    def get_parameters(self, config):
        return Utils.get_params(self.net)

    def write_state(state, path):
        with open(path, "wb") as f:
            dill.dump(state, f)

    def fit(self, parameters, config, average_probabilities=None):
        # print(f"Node {self.cid} received {average_probabilities}")
        Utils.set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])
        train_loader = Utils.get_dataloader(
            self.fed_dir,
            self.cid,
            batch_size=config["batch_size"],
            workers=num_workers,
            dataset=self.dataset_name,
            partition="train",
        )

        loaded_privacy_engine = None
        loaded_privacy_engine_regularization = None

        # If we already used this client we need to load the state regarding
        # the private model
        if os.path.exists(
            f"{self.fed_dir}/privacy_engine_{self.cid}.pkl"
        ) and os.path.exists(
            f"{self.fed_dir}/privacy_engine_regularization_{self.cid}.pkl"
        ):
            with open(f"{self.fed_dir}/privacy_engine_{self.cid}.pkl", "rb") as file:
                loaded_privacy_engine = dill.load(file)
            with open(
                f"{self.fed_dir}/privacy_engine_regularization_{self.cid}.pkl", "rb"
            ) as file:
                loaded_privacy_engine_regularization = dill.load(file)
        else:
            self.train_parameters.DPL_lambda = 0

            # compute the disparity of the training dataset
            disparities = []
            for target in range(0, 1):
                for sv in range(0, 1):
                    disparities.append(
                        RegularizationLoss().compute_violation_with_argmax(
                            predictions_argmax=train_loader.dataset.targets,
                            sensitive_attribute_list=train_loader.dataset.sensitive_features,
                            current_target=target,
                            current_sensitive_feature=sv,
                        )
                    )
            max_disparity_train = np.mean(disparities)

        (
            private_net,
            private_optimizer,
            train_loader,
            privacy_engine,
        ) = Utils.create_private_model(
            model=self.net,
            epsilon=self.train_parameters.epsilon,
            original_optimizer=self.optimizer,
            train_loader=train_loader,
            epochs=self.train_parameters.epochs,
            delta=self.delta,
            MAX_GRAD_NORM=self.clipping,
            batch_size=self.train_parameters.batch_size,
            noise_multiplier=self.train_parameters.noise_multiplier,
            accountant=loaded_privacy_engine,
        )
        private_net.to(self.train_parameters.device)

        private_model_regularization = None
        private_optimizer_regularization = None

        if self.train_parameters.DPL:
            (
                private_model_regularization,
                private_optimizer_regularization,
                _,
                privacy_engine_regularization,
            ) = Utils.create_private_model(
                model=self.model_regularization,
                epsilon=self.train_parameters.epsilon,
                original_optimizer=self.optimizer_regularization,
                train_loader=train_loader,
                epochs=self.train_parameters.epochs,
                delta=self.delta,
                MAX_GRAD_NORM=self.clipping,
                batch_size=self.train_parameters.batch_size,
                noise_multiplier=self.train_parameters.noise_multiplier,
                accountant=loaded_privacy_engine_regularization,
            )
            private_model_regularization.to(self.train_parameters.device)

        gc.collect()

        all_metrics = []
        all_losses = []
        for epoch in range(0, self.train_parameters.epochs):
            print(f"Training Epoch: {epoch}")
            metrics = Learning.train_private_model(
                train_parameters=self.train_parameters,
                model=private_net,
                model_regularization=private_model_regularization,
                optimizer=private_optimizer,
                optimizer_regularization=private_optimizer_regularization,
                train_loader=train_loader,
                test_loader=None,
                current_epoch=epoch,
                node_id=self.cid,
                average_probabilities=average_probabilities,
            )
            all_metrics.append(metrics)
            all_losses.append(metrics["Train Loss"])

        Utils.set_params(self.net, Utils.get_params(private_net))

        # We need to store the state of the privacy engine and all the
        # details about the private training
        with open(f"{self.fed_dir}/privacy_engine_{self.cid}.pkl", "wb") as f:
            dill.dump(privacy_engine.accountant, f)
        with open(
            f"{self.fed_dir}/privacy_engine_regularization_{self.cid}.pkl", "wb"
        ) as f:
            dill.dump(privacy_engine_regularization.accountant, f)

        with open(f"{self.fed_dir}/DPL_lambda_{self.cid}.pkl", "wb") as f:
            dill.dump(self.train_parameters.DPL_lambda, f)

        (
            predictions,
            sensitive_attributes,
            possible_targets,
            possible_sensitive_attributes,
        ) = Learning.test_prediction(
            model=private_net,
            test_loader=train_loader,
            train_parameters=self.train_parameters,
            current_epoch=None,
        )
        print(f"Computed metrics on train data on node {self.cid}")
        probabilities, counters = RegularizationLoss.compute_probabilities(
            predictions=predictions,
            sensitive_attribute_list=sensitive_attributes,
            device=self.train_parameters.device,
            possible_sensitive_attributes=possible_sensitive_attributes,
            possible_targets=possible_targets,
        )
        print(f"Computed probabilities on node {self.cid}")

        del private_net
        if private_model_regularization:
            del private_model_regularization
        gc.collect()

        # Return local model and statistics
        return (
            Utils.get_params(self.net),
            len(train_loader.dataset),
            {
                "train_losses": all_losses,
                "train_loss": all_metrics[-1]["Train Loss"],
                "train_loss_with_regularization": all_metrics[-1][
                    "Train Loss + Regularizaion"
                ],
                "train_accuracy": all_metrics[-1]["Train Accuracy"],
                "epsilon": privacy_engine.accountant.get_epsilon(self.delta),
                "probabilities": probabilities,
                "cid": self.cid,
                "targets": possible_targets,
                "sensitive_attributes": possible_sensitive_attributes,
                "Disparity Train": all_metrics[-1]["Max Disparity Train"],
                "Lambda": self.train_parameters.DPL_lambda,
                "counters": counters,
                "Max Disparity Train Before Local Epoch": all_metrics[0][
                    "Max Disparity Train Before Local Epoch"
                ],
                "Max Disparity Dataset": max_disparity_train,
            },
        )

    def evaluate(self, parameters, config):
        print(f"CALLING EVALUATE FUNCTION on node {self.cid}")
        Utils.set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])

        valloader = Utils.get_dataloader(
            self.fed_dir,
            self.cid,
            batch_size=self.train_parameters.batch_size,
            workers=num_workers,
            dataset=self.dataset_name,
            partition="val" if self.train_parameters.sweep else "test",
        )

        print(f"USING PARTITION {'val' if self.train_parameters.sweep else 'test'}")

        # Send model to device
        self.net.to(self.train_parameters.device)

        # Evaluate
        (
            test_loss,
            accuracy,
            f1score,
            precision,
            recall,
            max_disparity,
        ) = Learning.test(
            model=self.net,
            test_loader=valloader,
            train_parameters=self.train_parameters,
            current_epoch=None,
        )

        (
            predictions,
            sensitive_attributes,
            possible_targets,
            possible_sensitive_attributes,
        ) = Learning.test_prediction(
            model=self.net,
            test_loader=valloader,
            train_parameters=self.train_parameters,
            current_epoch=None,
        )
        probabilities, counters = RegularizationLoss.compute_probabilities(
            predictions=predictions,
            sensitive_attribute_list=sensitive_attributes,
            device=self.train_parameters.device,
            possible_sensitive_attributes=possible_sensitive_attributes,
            possible_targets=possible_targets,
        )

        self.net.to("cpu")
        gc.collect()

        if self.train_parameters.sweep:
            metrics = {
                "validation_accuracy": float(accuracy),
                "max_disparity_validation": float(max_disparity),
                "validation_loss": test_loss,
                "probabilities": probabilities,
                "cid": self.cid,
                "counters": counters,
            }
        else:
            metrics = {
                "test_accuracy": float(accuracy),
                "max_disparity_test": float(max_disparity),
                "test_loss": test_loss,
                "probabilities": probabilities,
                "cid": self.cid,
                "counters": counters,
            }

        # Return statistics
        return (
            float(test_loss),
            len(valloader.dataset),
            metrics,
        )
