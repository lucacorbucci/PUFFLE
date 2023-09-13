import random
import sys
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from DPL.RegularizationLoss import RegularizationLoss
from DPL.Utils.train_parameters import TrainParameters
from DPL.Utils.utils import Utils


class Learning:
    @staticmethod
    def train_loop(
        train_parameters: TrainParameters,
        model: torch.nn.Module,
        model_regularization: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        optimizer_regularization: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
    ) -> torch.nn.Module:
        """Train loop function for the model.
        For each epoch, first train the model on
        the training set, then test it on the test set.

        Args:
            train_parameters (TrainParameters): the parameters of the training
            model (torch.nn.Module): the model to train
            model_regularization (torch.nn.Module): if DPL is True, the model
                used to compute the regularization term
            optimizer (torch.optim.Optimizer): the optimizer used to train the
                model
            optimizer_regularization (torch.optim.Optimizer): the optimizer used
                to train the model_regularization if DPL is True
            train_loader (torch.utils.data.DataLoader): the training set
            test_loader (torch.utils.data.DataLoader): the test set

        Returns:
            torch.nn.Module: the model trained
        """

        for epoch in range(0, train_parameters.epochs):
            if train_parameters.private:
                Learning.train_private_model(
                    train_parameters=train_parameters,
                    model=model,
                    model_regularization=model_regularization,
                    optimizer=optimizer,
                    optimizer_regularization=optimizer_regularization,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    current_epoch=epoch,
                )
            else:
                Learning.train_model(
                    train_parameters=train_parameters,
                    model=model,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    current_epoch=epoch,
                )

            (
                test_loss,
                accuracy,
                f1score,
                precision,
                recall,
                max_disparity_test,
            ) = Learning.test(
                model=model,
                test_loader=test_loader,
                train_parameters=train_parameters,
                current_epoch=epoch,
            )

        return model

    @staticmethod
    def train_private_model(
        train_parameters: TrainParameters,
        model: torch.nn.Module,
        model_regularization: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        optimizer_regularization: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        current_epoch: int,
        node_id: int,
        average_probabilities: dict,
    ):
        """This function is used to train the private model.
        If we want to use DPL, then we need two models: one for the
        classic training from which we get the loss and one for the
        DPL computation.
        In this case, to inject the DPL in the training, we need to
        sum the per sample gradients of the two models. We can't just
        sum the losses.

        Args:
            train_parameters (TrainParameters): The parameters of the training
            model (torch.nn.Module): the model to train
            model_regularization (torch.nn.Module): the model used to compute
                the regularization term. None if DPL is False
            optimizer (torch.optim.Optimizer): the optimizer used to train the
                model
            optimizer_regularization (torch.optim.Optimizer): the optimizer used
                to train the model_regularization if DPL is True
            train_loader (torch.utils.data.DataLoader): the training set
            test_loader (torch.utils.data.DataLoader): the test set
            current_epoch (int): the current epoch

        Raises:
            ValueError: if model_regularization is None and DPL is True
        """
        criterion = nn.CrossEntropyLoss()
        criterion_regularization = RegularizationLoss()
        losses = []
        losses_with_regularization = []
        total_correct = 0
        total = 0
        model.train()
        if model_regularization:
            model_regularization.train()

        regularization_term = None

        MAX_PHYSICAL_BATCH_SIZE = 512
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer,
        ) as memory_safe_data_loader:
            for batch_index, (data, sensitive_feature, target) in enumerate(
                memory_safe_data_loader, 0
            ):
                regularization_term = None
                # If we use DPL, we need to synchronize the two models
                # before each batch because we are only updating the
                # weights of the model and not the weights of the
                # model_regularization
                if model_regularization is not None:
                    Utils.sync_models(model_regularization, model)

                optimizer.zero_grad()
                target = target.to(train_parameters.device)
                data = data.to(train_parameters.device)
                sensitive_feature = sensitive_feature.to(train_parameters.device)

                if (
                    train_parameters.DPL
                    and train_parameters.private
                    and not model_regularization
                ):
                    raise ValueError(
                        "model_regularization can't be None if DPL and private are True"
                    )

                # If DPL is True, we need to compute the regularization term
                # and add it to the loss. The computation of the regularization
                # term is done as in the classic training, but with the model
                # that is used to compute the regularization term.
                if train_parameters.DPL and model_regularization:
                    output_regularization = model_regularization(data)
                    fairness_violation = Learning.compute_regularization_term(
                        data=data,
                        target=target,
                        sensitive_feature=sensitive_feature,
                        train_parameters=train_parameters,
                        criterion_regularization=criterion_regularization,
                        outputs=output_regularization,
                        average_probabilities=average_probabilities,
                    )

                    if train_parameters.wandb_run:
                        train_parameters.wandb_run.log(
                            {
                                "Fairness Violation Batch": fairness_violation.item(),
                            }
                        )

                    regularization_term = (
                        train_parameters.DPL_lambda * fairness_violation
                    )
                    regularization_term.backward()

                outputs = model(data)

                # The classic loss is multiplied with (1 - lambda) and then we
                # compute the backward pass on it.
                tmp_loss = criterion(outputs, target)
                loss = (1 - train_parameters.DPL_lambda) * tmp_loss
                losses.append(loss.item())
                losses_with_regularization.append((regularization_term + loss).item())
                loss.backward()
                # If we use DPL, we need to sum the per sample gradients of the
                # two models
                if regularization_term and train_parameters.DPL_lambda > 0:
                    for p1, p2 in zip(
                        model.parameters(), model_regularization.parameters()
                    ):
                        p1.grad_sample += p2.grad_sample

                optimizer.step()

                # We also log to wandb the loss of this batch and the
                # norm of the gradients
                if train_parameters.wandb_run:
                    train_parameters.wandb_run.log(
                        {
                            "Loss Batch": loss.item(),
                            "Gradient Norm": Utils.get_summed_grad(
                                model=model,
                                batch_size=len(target),
                            ),
                        }
                    )

                optimizer.zero_grad()
                if optimizer_regularization:
                    optimizer_regularization.zero_grad()

                # Compute the total number of correct predictions
                # and store the total number of predictions
                # We will need these information to compute the
                # accuracy of the model and to compute the maximum disparity
                # of the model after the end of this epoch
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == target).float().sum()
                total_correct += correct
                total += target.size(0)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            train_loss = np.mean(losses)
            train_loss_with_regularization = np.mean(losses_with_regularization)

            accuracy = total_correct / total
            max_disparity_train = criterion_regularization.violation_with_dataset(
                model, train_loader, train_parameters.device, average_probabilities=average_probabilities, test="TRAINING", train_parameters=train_parameters,
            )
            if train_parameters.wandb_run:
                train_parameters.wandb_run.log(
                    {
                        "epoch": current_epoch,
                        f"Train Loss node {node_id}": train_loss,
                        "Train Accuracy": accuracy,
                        "Max Disparity Train": max_disparity_train,
                    }
                )
            return {
                "epoch": current_epoch,
                "Train Loss": train_loss,
                "Train Loss + Regularizaion": train_loss_with_regularization,
                "Train Accuracy": accuracy,
                "Max Disparity Train": max_disparity_train,
            }

    @staticmethod
    def test(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        train_parameters: TrainParameters,
        current_epoch: int,
        set_name: str = "test set",
    ) -> Tuple[float, float, float, float, float, float]:
        """Test the model on the test set computing the
        accuracy and also the maximum disparity of the model.

        Args:
            model (torch.nn.Module): The model we want to test
            test_loader (torch.utils.data.DataLoader): the test dataset
            train_parameters (TrainParameters): the parameters used for the training
            current_epoch (int): the current epoch
            set_name (str, optional): name of the dataset used for the test.
                Defaults to "test set".
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        total = 0
        y_pred = []
        y_true = []
        losses = []
        predictions = []
        colors = []

        with torch.no_grad():
            for data, color, target in test_loader:
                data, target = data.to(train_parameters.device), target.to(
                    train_parameters.device
                )
                output = model(data)
                total += target.size(0)
                test_loss = criterion(output, target).item()
                losses.append(test_loss)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                y_pred.extend(pred)
                y_true.extend(target)
                predictions.extend(pred)
                colors += [item.item() for item in color]

        predictions = [value.item() for item in predictions for value in item]

        counter_predictions = defaultdict(list)
        counter_true_predictions = defaultdict(list)

        for prediction, color, true_value in zip(predictions, colors, y_true):
            counter_predictions[color].append(prediction)
            counter_true_predictions[color].append(true_value.item())

        for sensitive_attribute in counter_predictions.keys():
            cm = confusion_matrix(
                counter_true_predictions[sensitive_attribute],
                counter_predictions[sensitive_attribute],
            )
            tn, fp, fn, tp = cm.ravel()
            if train_parameters.wandb_run:
                train_parameters.wandb_run.log(
                    {
                        "epoch": current_epoch,
                        f"true negatives {sensitive_attribute}": tn,
                        f"false positives {sensitive_attribute}": fp,
                        f"false negatives {sensitive_attribute}": fn,
                        f"true positives {sensitive_attribute}": tp,
                    }
                )

        criterion_regularization = RegularizationLoss()

        max_disparity_test = criterion_regularization.violation_with_dataset(
            model, test_loader, train_parameters.device, test="TRUEEEE"
        )

        test_loss = np.mean(losses)
        accuracy = correct / total

        y_true = [item.item() for item in y_true]
        y_pred = [item.item() for item in y_pred]

        f1score = f1_score(y_true, y_pred, average="macro")
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")

        print(
            f"Performance on {set_name}: loss: {test_loss}, Accuracy: {accuracy}, \
            Max disparity Test: {max_disparity_test}"
        )

        if train_parameters.wandb_run:
            train_parameters.wandb_run.log(
                {
                    "test_loss": test_loss,
                    "test_accuracy": accuracy,
                    "epoch": current_epoch,
                    "max_disparity_test": max_disparity_test,
                    "f1_score": f1score,
                    "precision": precision,
                    "recall": recall,
                }
            )

        return (
            test_loss,
            accuracy,
            f1score,
            precision,
            recall,
            max_disparity_test,
        )

    @staticmethod
    def test_prediction(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        train_parameters: TrainParameters,
        current_epoch: int,
        set_name: str = "test set",
    ) -> Tuple[float, float, float, float, float, float]:
        """Test the model on the test set computing the
        accuracy and also the maximum disparity of the model.

        Args:
            model (torch.nn.Module): The model we want to test
            test_loader (torch.utils.data.DataLoader): the test dataset
            train_parameters (TrainParameters): the parameters used for the training
            current_epoch (int): the current epoch
            set_name (str, optional): name of the dataset used for the test.
                Defaults to "test set".
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        total = 0
        y_pred = []
        y_true = []
        losses = []
        predictions = []
        sensitive_attributes = []

        with torch.no_grad():
            for data, sensitive_attribute, target in test_loader:
                data, target = data.to(train_parameters.device), target.to(
                    train_parameters.device
                )
                output = model(data)
                total += target.size(0)
                test_loss = criterion(output, target).item()
                losses.append(test_loss)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                y_pred.extend(pred)
                y_true.extend(target)
                predictions.append(output)
                sensitive_attributes += [item.item() for item in sensitive_attribute]

        concatenated_predictions = torch.cat(predictions, dim=0)

        return (
            concatenated_predictions,
            torch.tensor(sensitive_attributes),
            set([item.item() for item in y_true]),
            set(sensitive_attributes),
        )

    @staticmethod
    def compute_regularization_term(
        data: torch.utils.data.DataLoader,
        target: torch.tensor,
        sensitive_feature: torch.tensor,
        criterion_regularization: RegularizationLoss,
        train_parameters: TrainParameters,
        outputs: torch.tensor,
        average_probabilities=dict,
    ) -> torch.tensor:
        """This function computes the regularization term on the training data
        passed as parameter.

        Args:
            data (torch.utils.data.DataLoader): the dataset on which the regularization
                term is computed
            target (torch.tensor): the targets of the data we pass as parameter
            sensitive_feature (torch.tensor): the corresponding sensitive features
            criterion_regularization (RegularizationLoss): the regularization criterion
            train_parameters (TrainParameters): the parameters used for the training
            outputs (torch.tensor): Output of the model

        Returns:
            fairness_violation (_type_): the fairness violation computed on the data
                This does not include the multiplication with the Lambda
                and the Backward pass.
        """
        possible_targets = set([item.item() for item in target])
        possible_sensitive_attributes = set([item.item() for item in sensitive_feature])
        fairness_violation = criterion_regularization(
            sensitive_attribute_list=sensitive_feature,
            device=train_parameters.device,
            predictions=outputs,
            possible_sensitive_attributes=list(possible_sensitive_attributes),
            possible_targets=list(possible_targets),
            train_parameters=train_parameters,
            wandb_run=train_parameters.wandb_run,
            average_probabilities=average_probabilities,
        )

        return fairness_violation
