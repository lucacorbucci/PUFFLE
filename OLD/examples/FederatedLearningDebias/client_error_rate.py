import copy
import gc
import json
import logging
import os
import warnings
from pathlib import Path

import dill
import flwr as fl
import numpy as np
import ray
import torch
from Utils.model_utils import ModelUtils
from Utils.train_parameters import TrainParameters
from Utils.utils import Utils
from flwr.common.typing import Scalar
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier

from FairReg.Learning.learning import Learning
from FairReg.Regularization.ErrorRateRegularizationLoss import ErrorRateRegularizationLoss


class FlowerClientErrorRate(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        fed_dir_data: str,
        dataset_name: str,
        clipping: float,
        lr: float,
        train_parameters: TrainParameters,
        client_generator,
    ):
        logging.info(f"Node {cid} is initializing...")
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        self.train_parameters = copy.deepcopy(train_parameters)
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        print(f"Client {self.cid} is using the directory {self.fed_dir}")
        self.properties: dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.dataset_name = dataset_name
        self.clipping = clipping
        self.lr = lr
        self.client_generator = client_generator
        self.net = ModelUtils.get_model(dataset_name, device=self.train_parameters.device)
        self.optimizer = self.get_optimizer(model=self.net)

        if self.train_parameters.regularization:
            self.model_regularization = ModelUtils.get_model(
                self.dataset_name,
                device=self.train_parameters.device,
            )
            self.optimizer_regularization = self.get_optimizer(model=self.model_regularization)

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
        is_tunable = True if self.train_parameters.regularization_mode == "tunable" else False
        current_fl_round = config["server_round"]
        random_generator = np.random.default_rng(seed=[int(self.client_generator.random() * 2**32), current_fl_round])
        seed = int(random_generator.random() * 2**32)
        Utils.seed_everything(seed)
        self.train_parameters.fed_dir = self.fed_dir

        if os.path.exists(f"{self.fed_dir}/avg_proba.pkl"):
            with open(f"{self.fed_dir}/avg_proba.pkl", "rb") as file:
                average_probabilities = dill.load(file)
                print("Average probabilities loaded: ", average_probabilities)

        Utils.set_params(self.net, parameters)

        with open(f"{self.fed_dir}/counter_sampling.pkl", "rb") as f:
            counter_sampling = dill.load(f)
            self.sampling_frequency = counter_sampling[str(self.cid)]

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
                self.sampling_frequency
                * self.train_parameters.epochs
                * len(train_loader)
                * (len(self.train_parameters.unprivileged_group) + len(self.train_parameters.privileged_group))
                if self.train_parameters.global_computation
                else 1
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

            if os.path.exists(f"{self.fed_dir}/privacy_engine_regularization_{self.cid}.pkl"):
                with open(f"{self.fed_dir}/privacy_engine_regularization_{self.cid}.pkl", "rb") as file:
                    loaded_privacy_engine_regularization = dill.load(file)
        else:
            # If it is the first time that we use this client we use a Lambda = 0
            # because in the first round the model will be random and so the predictions
            # so the error rate will be 0 and therefore we can have Lambda = 0.
            # This is just the Lambda that we will use in the first batch. Then we
            # will update it based on our classic algorithm.
            if self.train_parameters.regularization_mode == "tunable":
                self.train_parameters.regularization_lambda = 0
            first_round = True

        if self.train_parameters.epsilon is None:
            self.noise_multiplier = 0
            self.original_epsilon = None
        else:
            if os.path.exists(f"{self.fed_dir}/noise_level_{self.cid}.pkl"):
                with open(f"{self.fed_dir}/noise_level_{self.cid}.pkl", "rb") as file:
                    self.noise_multiplier = dill.load(file)
                    self.original_epsilon = self.train_parameters.epsilon
                    self.train_parameters.epsilon = None
            else:
                # We compute the noise corresponding to the epsilon defined
                # as parameter in the TrainParameter passed to the client
                noise = self.get_noise(dataset=train_loader)
                with open(f"{self.fed_dir}/noise_level_{self.cid}.pkl", "wb") as file:
                    dill.dump(noise, file)
                self.noise_multiplier = noise
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
            noise_multiplier=self.noise_multiplier,
            accountant=loaded_privacy_engine,
        )
        private_net.to(self.train_parameters.device)

        private_model_regularization = None
        private_optimizer_regularization = None

        if os.path.exists(f"{self.train_parameters.fed_dir}/sum_counters.pkl"):
            with open(f"{self.train_parameters.fed_dir}/sum_counters.pkl", "rb") as file:
                sum_counters = dill.load(file)
        else:
            sum_counters = None
            first_round = True

        # Use the model sent by the server to compute the error rate
        # before the local training
        max_error_rate_train_before_local_epoch = ErrorRateRegularizationLoss().violation_with_dataset(
            model=private_net,
            dataset=train_loader,
            device=self.train_parameters.device,
            average_probabilities=average_probabilities,
            privileged_group=self.train_parameters.privileged_group,
            unprivileged_group=self.train_parameters.unprivileged_group,
            # global_computation=self.train_parameters.global_computation,
        )

        # In the first round we want to start from Lambda = 0, if it is not the first
        # round we have several options to update Lambda: we can start from a fixed
        # value, we can start from a value that depends on the target error rate
        # and on the error rate of the training dataset
        if not first_round and self.train_parameters.target and is_tunable:
            # if max_error_rate_train_before_local_epoch is None:
            #     self.train_parameters.regularization_lambda = 0
            # else:
            #     self.train_parameters.regularization_lambda = (
            #         self.compute_starting_lambda_with_error_rate(
            #             error_rate_training=max_error_rate_train_before_local_epoch,
            #         )
            #     )
            max_error_rate_train_before_local_epoch = max_error_rate_train_before_local_epoch.item()
            self.train_parameters.regularization_lambda = np.clip(
                (
                    -(self.train_parameters.target - max_error_rate_train_before_local_epoch)
                    / self.train_parameters.target
                ),
                0,
                1,
            )
        print(
            "CID",
            self.cid,
            " - Amount of samples: ",
            len(train_loader.dataset),
            " - Starting Lambda: ",
            self.train_parameters.regularization_lambda,
            "error rate: ",
            max_error_rate_train_before_local_epoch,
        )

        if self.train_parameters.regularization:
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
                noise_multiplier=self.noise_multiplier,
                accountant=loaded_privacy_engine_regularization,
            )
            private_model_regularization.to(self.train_parameters.device)

        gc.collect()

        all_metrics = []
        all_losses = []
        history_lambda = []
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
                current_fl_round=current_fl_round,
                node_id=self.cid,
                average_probabilities=average_probabilities,
                sigma_update_lambda=sigma_update_lambda,
            )

            metrics["Max Error Rate Train Before Local Epoch"] = max_error_rate_train_before_local_epoch

            history_lambda.extend(metrics["history_lambda"])
            all_metrics.append(metrics)
            all_losses.append(metrics["Train Loss"])

        Utils.set_params(self.net, Utils.get_params(private_net))

        # We need to store the state of the privacy engine and all the
        # details about the private training
        with open(f"{self.fed_dir}/privacy_engine_{self.cid}.pkl", "wb") as f:
            dill.dump(privacy_engine.accountant, f)
        if self.train_parameters.regularization:
            with open(f"{self.fed_dir}/privacy_engine_regularization_{self.cid}.pkl", "wb") as f:
                dill.dump(privacy_engine_regularization.accountant, f)

            with open(f"{self.fed_dir}/regularization_lambda_{self.cid}.pkl", "wb") as f:
                dill.dump(self.train_parameters.regularization_lambda, f)

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

        predictions_argmax = list(torch.argmax(predictions, dim=1))

        accuracy_per_group = self.compute_accuracy_per_group(
            predictions=predictions_argmax,
            sensitive_attributes=sensitive_attributes,
            y_true=y_true,
        )
        error_rate_per_group = self.compute_error_rate_per_group(
            predictions=predictions_argmax,
            sensitive_attributes=sensitive_attributes,
            y_true=y_true,
        )

        probabilities, counters = ErrorRateRegularizationLoss.compute_probabilities(
            predictions=predictions,
            sensitive_attribute_list=sensitive_attributes,
            device=self.train_parameters.device,
            possible_sensitive_attributes=possible_sensitive_attributes,
            possible_targets=possible_targets,
            # train_parameters=self.train_parameters,
            true_targets=y_true,
            privileged_group=self.train_parameters.privileged_group,
            unprivileged_group=self.train_parameters.unprivileged_group,
        )

        counters_no_noise = copy.deepcopy(counters)

        # compute the noise that I have to add to the counters to ensure we
        # guarantee train_parameters.epsilon_statistics
        if self.train_parameters.epsilon_statistics is not None:
            sampling_ratio = 1
            iterations = (
                self.sampling_frequency * 2 * len(possible_sensitive_attributes)
            )  # we multiply by 2 because every time we send two values
            noise_statistics = get_noise_multiplier(
                target_epsilon=self.train_parameters.epsilon_statistics,
                target_delta=self.delta,
                sample_rate=sampling_ratio,
                steps=iterations,
                accountant="rdp",
            )

            for key in probabilities.keys():
                if len(key) > 1:
                    probabilities[key] += Utils.get_noise(mechanism_type="gaussian", sigma=noise_statistics)
                    # we do not want to have negative counters
                    if probabilities[key] < 0:
                        probabilities[key] = 0

        else:
            noise_statistics = None

        # Compute the final epsilon by summing the three epsilons
        # that we can have in the methodology. We can do this because
        # all the epsilon are RDP
        if self.original_epsilon:
            final_epsilon = (
                self.original_epsilon
                + (self.train_parameters.epsilon_lambda if self.train_parameters.epsilon_lambda is not None else 0)
                + (
                    self.train_parameters.epsilon_statistics
                    if self.train_parameters.epsilon_statistics is not None
                    else 0
                )
            )
        else:
            final_epsilon = float("inf")

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
                "train_loss_with_regularization": all_metrics[-1]["Train Loss + Regularizaion"],
                "train_accuracy": all_metrics[-1]["Train Accuracy"],
                "epsilon": final_epsilon,
                "delta": final_delta,
                "probabilities": probabilities,
                "cid": self.cid,
                "targets": possible_targets,
                "sensitive_attributes": possible_sensitive_attributes,
                "Lambda": self.train_parameters.regularization_lambda,
                "counters": counters,
                "counters_no_noise": counters_no_noise,
                "history_lambda": history_lambda,
                "accuracy_per_group": accuracy_per_group,
                "error_rate_per_group": error_rate_per_group,
                "Error Rate Train": all_metrics[-1]["Max Unfairness Train"]
                if all_metrics[-1]["Max Unfairness Train"] != None
                else 0,
            },
        )

    def compute_accuracy_per_group(self, predictions, sensitive_attributes, y_true):
        # given the possible sensitive attributes in sensitive_attributes list
        # we compute the accuracy for each possible sensitive attribute
        # and we return a dictionary with the accuracy for each group
        predictions = [item.item() if isinstance(item, torch.Tensor) else item for item in predictions]
        sensitive_attributes = [
            item.item() if isinstance(item, torch.Tensor) else item for item in sensitive_attributes
        ]
        y_true = [item.item() if isinstance(item, torch.Tensor) else item for item in y_true]
        accuracy_per_group = {}
        for sensitive_attribute in sensitive_attributes:
            accuracy_per_group[sensitive_attribute] = 0
            total = 0
            correct = 0
            for i in range(len(predictions)):
                if sensitive_attribute == sensitive_attributes[i]:
                    total += 1
                    if predictions[i] == y_true[i]:
                        correct += 1
            accuracy_per_group[sensitive_attribute] = correct / total
            accuracy_per_group[sensitive_attribute] = correct / total
        return accuracy_per_group

    def compute_error_rate_per_group(self, predictions, sensitive_attributes, y_true):
        predictions = [item.item() if isinstance(item, torch.Tensor) else item for item in predictions]
        sensitive_attributes = [
            item.item() if isinstance(item, torch.Tensor) else item for item in sensitive_attributes
        ]
        possible_sensitive_attributes = list(set(sensitive_attributes))
        y_true = [item.item() if isinstance(item, torch.Tensor) else item for item in y_true]

        predictions = np.array(predictions)
        y_true = np.array(y_true)
        sensitive_attributes = np.array(sensitive_attributes)

        error_rate_per_group = {}
        for sensitive_attribute in possible_sensitive_attributes:
            # take the indexes of the elements that have the sensitive attribute
            indexes = np.where(sensitive_attributes == sensitive_attribute)
            # take the predictions of the elements that have the sensitive attribute
            predictions_sensitive = predictions[indexes]
            # take the true values of the elements that have the sensitive attribute
            y_true_sensitive = y_true[indexes]
            count_errors = 0
            total = 0
            count_errors = np.sum(predictions_sensitive != y_true_sensitive)
            total = len(predictions_sensitive)
            error_rate_per_group[sensitive_attribute] = (count_errors, total)

        # error_rate_per_group = {}
        # for sensitive_attribute in sensitive_attributes:
        #     count_errors = 0
        #     total = 0
        #     for i in range(len(predictions)):
        #         if sensitive_attribute == sensitive_attributes[i]:
        #             total += 1
        #             if predictions[i] != y_true[i]:
        #                 count_errors += 1
        #     error_rate_per_group[sensitive_attribute] = count_errors / total
        return error_rate_per_group

    def evaluate(self, parameters, config):
        if os.path.exists(f"{self.fed_dir}/avg_proba.pkl"):
            with open(f"{self.fed_dir}/avg_proba.pkl", "rb") as file:
                average_probabilities = dill.load(file)
        else:
            average_probabilities = None
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

        # Send model to device
        self.net.to(self.train_parameters.device)

        # Evaluate
        (
            test_loss,
            accuracy,
            f1score,
            precision,
            recall,
            max_error_rate,
            _,
            _,
            _,
        ) = Learning.test(
            model=self.net,
            test_loader=dataset,
            train_parameters=self.train_parameters,
            current_epoch=None,
            average_probabilities=average_probabilities,
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
        ) = ErrorRateRegularizationLoss.compute_probabilities(
            predictions=predictions,
            sensitive_attribute_list=sensitive_attributes,
            device=self.train_parameters.device,
            possible_sensitive_attributes=possible_sensitive_attributes,
            possible_targets=possible_targets,
            true_targets=y_true,
            privileged_group=self.train_parameters.privileged_group,
            unprivileged_group=self.train_parameters.unprivileged_group,
        )

        predictions_argmax = list(torch.argmax(predictions, dim=1))

        accuracy_per_group = self.compute_accuracy_per_group(
            predictions=predictions_argmax,
            sensitive_attributes=sensitive_attributes,
            y_true=y_true,
        )
        error_rate_per_group = self.compute_error_rate_per_group(
            predictions=predictions_argmax,
            sensitive_attributes=sensitive_attributes,
            y_true=y_true,
        )

        self.net.to("cpu")
        gc.collect()

        if self.train_parameters.sweep:
            metrics = {
                "validation_accuracy": float(accuracy),
                "max_error_rate_validation": float(max_error_rate) if max_error_rate != None else 0,
                "validation_loss": test_loss,
                "probabilities": probabilities,
                "cid": self.cid,
                "counters": counters,
                "f1_score": f1score,
                "accuracy_per_group": accuracy_per_group,
                "error_rate_per_group": error_rate_per_group,
            }
        else:
            metrics = {
                "test_accuracy": float(accuracy),
                "max_error_rate_test": float(max_error_rate) if max_error_rate != None else 0,
                "test_loss": test_loss,
                "probabilities": probabilities,
                "cid": self.cid,
                "counters": counters,
                "f1_score": f1score,
                "accuracy_per_group": accuracy_per_group,
                "error_rate_per_group": error_rate_per_group,
            }

        # Return statistics
        return (
            float(test_loss),
            len(dataset.dataset),
            metrics,
        )

    def compute_starting_lambda_with_error_rate(self, error_rate_training: float):
        """
        This function computes the starting Lambda based on
        the error rate of the training dataset and the target error rate.
        Given a certain target error rate and the actual error rate of the training
        dataset, what we do is to compute the difference between the two values.
        If the difference is positive, it means that we want to use a Lambda = 0.
        If the difference is negative then we can use the difference as a Lambda but
        instead of using it directly we have to rescale it in the range [0, 1].

        We need to rescale it because when we compute the difference we can have a negative
        value that is in the range [0, 1 - target_error rate]. However, we want to use the
        dellta in the range [0, 1] so we have to rescale it. To rescale it we have
        a function and we have to use as old min value 0 and as old max value 1 - target_error_rate.

        Even if it seems that this is a small detail, it is important to rescale the lambda
        in the correct way because starting with a wrong lambda can lead to a wrong
        regularization and so to a wrong model. Even if the lambda is updated during the
        training, the starting value is important.
        """

        delta = self.train_parameters.target - error_rate_training
        if delta > 0:
            return 0
        else:
            return Utils.rescale_lambda(
                value=abs(delta),
                old_min=0,
                old_max=1 - self.train_parameters.target,
                new_min=0,
                new_max=1,
            )

    def get_noise(self, dataset, target_epsilon=None):
        model_noise = ModelUtils.get_model(self.dataset_name, device=self.train_parameters.device)
        privacy_engine = PrivacyEngine(accountant="rdp")
        optimizer_noise = Utils.get_optimizer(model_noise, self.train_parameters, self.lr)
        (
            _,
            private_optimizer,
            _,
        ) = privacy_engine.make_private_with_epsilon(
            module=model_noise,
            optimizer=optimizer_noise,
            data_loader=dataset,
            epochs=self.sampling_frequency * self.train_parameters.epochs,
            target_epsilon=self.train_parameters.epsilon if target_epsilon is None else target_epsilon,
            target_delta=self.delta,
            max_grad_norm=self.clipping,
        )

        return private_optimizer.noise_multiplier
