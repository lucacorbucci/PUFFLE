import copy
import gc
import logging
import os
import warnings
from pathlib import Path

import dill
import flwr as fl
import numpy as np
import ray
import torch
from flwr.common.typing import Scalar
from opacus import PrivacyEngine
from opacus.accountants.rdp import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier

from DPL.ErrorRateRegularizationLoss import ErrorRateRegularizationLoss
from DPL.RegularizationLoss import RegularizationLoss
from DPL.Utils.model_utils import ModelUtils
from DPL.learning import Learning
from fl_puf.Utils.train_parameters import TrainParameters
from fl_puf.Utils.utils import Utils


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        fed_dir_data: str,
        dataset_name: str,
        clipping: float,
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

    def fit(self, parameters, config, average_probabilities=None):
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

        self.delta = (1 / len(train_loader.dataset)) / 3

        if self.train_parameters.epsilon_lambda is not None:
            # this is the sigma that we will use to compute the noise
            # that will be added to the Lambda
            sampling_ratio = 1 / len(train_loader)

            iterations = (
                self.train_parameters.sampling_frequency
                * self.train_parameters.epochs
                * len(train_loader)
            )
            sigma_update_lambda = get_noise_multiplier(
                target_epsilon=self.train_parameters.epsilon_lambda,
                target_delta=self.delta,
                sample_rate=sampling_ratio,
                steps=iterations,
                accountant="rdp",
            )
        else:
            sigma_update_lambda = None

        loaded_privacy_engine = None
        loaded_privacy_engine_regularization = None
        first_round = False

        # If we already used this client we need to load the state regarding
        # the privacy engine both for the classic model and for the model
        # used for the regularization
        if os.path.exists(f"{self.fed_dir}/privacy_engine_{self.cid}.pkl"):
            with open(f"{self.fed_dir}/privacy_engine_{self.cid}.pkl", "rb") as file:
                loaded_privacy_engine = dill.load(file)

            if os.path.exists(
                f"{self.fed_dir}/privacy_engine_regularization_{self.cid}.pkl"
            ):
                with open(
                    f"{self.fed_dir}/privacy_engine_regularization_{self.cid}.pkl", "rb"
                ) as file:
                    loaded_privacy_engine_regularization = dill.load(file)
        else:
            # If it is the first time that we use this client we use a Lambda = 0
            # because in the first round the model will be random and so the predictions
            # so the disparity will be 0 and therefore we can have Lambda = 0.
            # This is just the Lambda that we will use in the first batch. Then we
            # will update it based on our classic algorithm.
            self.train_parameters.DPL_lambda = 0
            first_round = True
        if self.train_parameters.starting_lambda_mode == "no_tuning":
            self.train_parameters.DPL_lambda = (
                self.train_parameters.starting_lambda_value
            )

        # compute the maximum disparity of the training dataset
        max_disparity_dataset = np.max(
            [
                RegularizationLoss().compute_violation_with_argmax(
                    predictions_argmax=train_loader.dataset.targets,
                    sensitive_attribute_list=train_loader.dataset.sensitive_features,
                    current_target=target,
                    current_sensitive_feature=sv,
                )
                for target in range(0, 1)
                for sv in range(0, 1)
            ]
        )

        if self.train_parameters.noise_multiplier is not None:
            print("noise multiplier is not none")
            noise = self.train_parameters.noise_multiplier
        else:
            if os.path.exists(f"{self.fed_dir}/noise_level_{self.cid}.pkl"):
                with open(f"{self.fed_dir}/noise_level_{self.cid}.pkl", "rb") as file:
                    self.train_parameters.noise_multiplier = dill.load(file)
                    noise = self.train_parameters.noise_multiplier
                    self.original_epsilon = self.train_parameters.epsilon
                    self.train_parameters.epsilon = None
            else:
                noise = self.get_noise(dataset=train_loader)
                with open(f"{self.fed_dir}/noise_level_{self.cid}.pkl", "wb") as file:
                    dill.dump(noise, file)
                self.train_parameters.noise_multiplier = noise
                self.original_epsilon = self.train_parameters.epsilon
                self.train_parameters.epsilon = None

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
            noise_multiplier=noise,
            accountant=loaded_privacy_engine,
        )
        private_net.to(self.train_parameters.device)

        private_model_regularization = None
        private_optimizer_regularization = None

        # Use the model sent by the server to compute the disparity
        # before the local training
        max_disparity_train_before_local_epoch = (
            RegularizationLoss().violation_with_dataset(
                private_net,
                train_loader,
                self.train_parameters.device,
            )
        )

        # In the first round we want to start from Lambda = 0, if it is not the first
        # round we have several options to update Lambda: we can start from a fixed
        # value, we can start from a value that depends on the target disparity
        # and on the disparity of the training dataset or we can use the average of the
        # disparities of the previous FL round
        if (
            not first_round
            and self.train_parameters.target
            and self.train_parameters.update_lambda
        ):
            if self.train_parameters.starting_lambda_mode == "fixed":
                self.train_parameters.DPL_lambda = (
                    self.train_parameters.starting_lambda_value
                )
            elif self.train_parameters.starting_lambda_mode == "avg":
                self.train_parameters.DPL_lambda = (
                    self.compute_starting_lambda_with_avg()
                )
            elif self.train_parameters.starting_lambda_mode == "disparity":
                self.train_parameters.DPL_lambda = self.compute_starting_lambda_with_disparity(
                    disparity_training=max_disparity_train_before_local_epoch,  # max_disparity_dataset,
                )
            else:
                raise ValueError(
                    f"Starting Lambda Mode not recognized, your value is: {self.train_parameters.starting_lambda_mode}"
                )

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
                sigma_update_lambda=sigma_update_lambda,
            )

            metrics[
                "Max Disparity Train Before Local Epoch"
            ] = max_disparity_train_before_local_epoch

            all_metrics.append(metrics)
            all_losses.append(metrics["Train Loss"])

        Utils.set_params(self.net, Utils.get_params(private_net))

        # We need to store the state of the privacy engine and all the
        # details about the private training
        with open(f"{self.fed_dir}/privacy_engine_{self.cid}.pkl", "wb") as f:
            dill.dump(privacy_engine.accountant, f)
        if self.train_parameters.DPL:
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
            y_true,
        ) = Learning.test_prediction(
            model=private_net,
            test_loader=train_loader,
            train_parameters=self.train_parameters,
            current_epoch=None,
        )
        (probabilities, counters) = RegularizationLoss.compute_probabilities(
            predictions=predictions,
            sensitive_attribute_list=sensitive_attributes,
            device=self.train_parameters.device,
            possible_sensitive_attributes=possible_sensitive_attributes,
            possible_targets=possible_targets,
            train_parameters=self.train_parameters,
        )

        (
            _,
            counters_error_rate,
        ) = ErrorRateRegularizationLoss.compute_probabilities(
            predictions=predictions,
            sensitive_attribute_list=sensitive_attributes,
            device=self.train_parameters.device,
            possible_sensitive_attributes=possible_sensitive_attributes,
            possible_targets=possible_targets,
            train_parameters=self.train_parameters,
            real_targets=y_true,
        )

        counters_no_noise = copy.deepcopy(counters)

        # compute the noise that I have to add to the counters to ensure we
        # guarantee train_parameters.epsilon_statistics
        if self.train_parameters.epsilon_statistics is not None:
            sampling_ratio = 1
            iterations = (
                self.train_parameters.sampling_frequency * 2
            )  # we multiply by 2 because every time we send two values
            noise_statistics = get_noise_multiplier(
                target_epsilon=self.train_parameters.epsilon_statistics,
                target_delta=self.delta,
                sample_rate=sampling_ratio,
                steps=iterations,
                accountant="rdp",
            )

            for key in counters.keys():
                if len(key) > 1:
                    counters[key] += Utils.get_noise(
                        mechanism_type="gaussian", sigma=noise_statistics
                    )
        else:
            noise_statistics = None

        # convert rdp to (epsilon, delta)-DP
        final_epsilon = (
            self.original_epsilon
            + (
                self.train_parameters.epsilon_lambda
                if self.train_parameters.epsilon_lambda is not None
                else 0
            )
            + (
                self.train_parameters.epsilon_statistics
                if self.train_parameters.epsilon_statistics is not None
                else 0
            )
        )
        final_delta = self.delta * 3

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
                # "epsilon": privacy_engine.accountant.get_epsilon(self.delta),
                "epsilon": final_epsilon,
                "delta": final_delta,
                "probabilities": probabilities,
                "cid": self.cid,
                "targets": possible_targets,
                "sensitive_attributes": possible_sensitive_attributes,
                "Disparity Train": all_metrics[-1]["Max Disparity Train"],
                "Lambda": self.train_parameters.DPL_lambda,
                "counters": counters,
                "counters_error_rate": counters_error_rate,
                "counters_no_noise": counters_no_noise,
                "Max Disparity Train Before Local Epoch": all_metrics[0][
                    "Max Disparity Train Before Local Epoch"
                ],
                "Max Disparity Dataset": max_disparity_dataset,
                "DPL_lambda": self.train_parameters.DPL_lambda,
            },
        )

    def evaluate(self, parameters, config):
        Utils.set_params(self.net, parameters)

        # Load data for this client and get trainloader
        num_workers = int(ray.get_runtime_context().get_assigned_resources()["CPU"])

        dataset = Utils.get_dataloader(
            self.fed_dir,
            self.cid,
            batch_size=self.train_parameters.batch_size,
            workers=num_workers,
            dataset=self.dataset_name,
            partition="train",
        )

        # compute the maximum disparity of the training dataset
        max_disparity_dataset = np.max(
            [
                RegularizationLoss().compute_violation_with_argmax(
                    predictions_argmax=dataset.dataset.targets,
                    sensitive_attribute_list=dataset.dataset.sensitive_features,
                    current_target=target,
                    current_sensitive_feature=sv,
                )
                for target in range(0, 1)
                for sv in range(0, 1)
            ]
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
            max_disparity,
        ) = Learning.test(
            model=self.net,
            test_loader=dataset,
            train_parameters=self.train_parameters,
            current_epoch=None,
        )

        (
            predictions,
            sensitive_attributes,
            possible_targets,
            possible_sensitive_attributes,
            y_true,
        ) = Learning.test_prediction(
            model=self.net,
            test_loader=dataset,
            train_parameters=self.train_parameters,
            current_epoch=None,
        )
        (
            probabilities,
            counters,
        ) = RegularizationLoss.compute_probabilities(
            predictions=predictions,
            sensitive_attribute_list=sensitive_attributes,
            device=self.train_parameters.device,
            possible_sensitive_attributes=possible_sensitive_attributes,
            possible_targets=possible_targets,
            train_parameters=self.train_parameters,
        )

        (
            _,
            counters_error_rate,
        ) = ErrorRateRegularizationLoss.compute_probabilities(
            predictions=predictions,
            sensitive_attribute_list=sensitive_attributes,
            device=self.train_parameters.device,
            possible_sensitive_attributes=possible_sensitive_attributes,
            possible_targets=possible_targets,
            train_parameters=self.train_parameters,
            real_targets=y_true,
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
                "counters_error_rate": counters_error_rate,
                "max_disparity_dataset": max_disparity_dataset,
                "f1_score": f1score,
            }
        else:
            metrics = {
                "test_accuracy": float(accuracy),
                "max_disparity_test": float(max_disparity),
                "test_loss": test_loss,
                "probabilities": probabilities,
                "cid": self.cid,
                "counters": counters,
                "counters_error_rate": counters_error_rate,
                "max_disparity_dataset": max_disparity_dataset,
                "f1_score": f1score,
            }

        # Return statistics
        return (
            float(test_loss),
            len(dataset.dataset),
            metrics,
        )

    def compute_starting_lambda_with_avg(self):
        """
        This function computes the staƒrting Lambda based on
        the average of the disparities of the previous FL round.
        """
        loaded_clients_list = []
        if os.path.exists(f"{self.fed_dir}/clients_last_round.pkl"):
            with open(f"{self.fed_dir}/clients_last_round.pkl", "rb") as file:
                loaded_clients_list = dill.load(file)
        lambda_list = []
        if loaded_clients_list:
            for client_cid in loaded_clients_list:
                if os.path.exists(f"{self.fed_dir}/DPL_lambda_{client_cid}.pkl"):
                    with open(
                        f"{self.fed_dir}/DPL_lambda_{client_cid}.pkl", "rb"
                    ) as file:
                        loaded_lambda = dill.load(file)
                        lambda_list.append(loaded_lambda)
        if lambda_list:
            return np.mean(lambda_list)
        return 0

    def compute_starting_lambda_with_disparity(self, disparity_training: float):
        """
        This function computes the starting Lambda based on
        the disparity of the training dataset and the target disparity.
        Given a certain target disparity and the actual disparity of the training
        dataset, what we do is to compute the difference between the two values.
        If the difference is positive, it means that we want to use a Lambda = 0.
        If the difference is negative then we can use the difference as a Lambda but
        instead of using it directly we have to rescale it in the range [0, 1].
        """
        delta = self.train_parameters.target - disparity_training
        if delta > 0:
            return 0
        else:
            return Utils.rescale_lambda(
                value=abs(delta),
                old_min=0,
                old_max=disparity_training,
                new_min=0,
                new_max=1,
            )

    def get_noise(self, dataset, target_epsilon=None):
        model_noise = ModelUtils.get_model(
            self.dataset_name, device=self.train_parameters.device
        )
        privacy_engine = PrivacyEngine(accountant="rdp")
        optimizer_noise = Utils.get_optimizer(
            model_noise, self.train_parameters, self.lr
        )
        (
            _,
            private_optimizer,
            _,
        ) = privacy_engine.make_private_with_epsilon(
            module=model_noise,
            optimizer=optimizer_noise,
            data_loader=dataset,
            epochs=self.train_parameters.sampling_frequency
            * self.train_parameters.epochs,
            target_epsilon=self.train_parameters.epsilon
            if target_epsilon is None
            else target_epsilon,
            target_delta=self.delta,
            max_grad_norm=self.clipping,
        )
        print(
            f"NOISE COMPUTE ON NODE {self.cid} is {private_optimizer.noise_multiplier}"
        )

        return private_optimizer.noise_multiplier
