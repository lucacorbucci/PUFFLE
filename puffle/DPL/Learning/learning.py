"""
This file implements the Learning class that is used to train the model
If you want to train the model using the unfairness mitigation through
Regularization and Differential Privacy, you need to use this class for the training
(or implement something similar to this one)
"""

import traceback
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from opacus.utils.batch_memory_manager import BatchMemoryManager
from sklearn.metrics import f1_score, precision_score, recall_score

from ..DPLUtils.regularization_config import RegularizationConfig
from ..DPLUtils.utils import Utils
from ..Regularization.RegularizationLoss import RegularizationLoss


def exp_lr_scheduler(initial_alpha, current_fl_round, decay_rate=0.001):
    """
    Decay learning rate by a factor of decay_rate every epoch.

    Args:
        initial_alpha (float): initial learning rate
        current_fl_round (int): the current fl round in which the client was selected
        decay_rate (float, optional): decay rate. Defaults to 0.1.
    """
    new_alpha = initial_alpha * decay_rate ** (current_fl_round + 1)
    return new_alpha


class Learning:
    @staticmethod
    def train_private_model(
        train_parameters: RegularizationConfig,
        model: torch.nn.Module,
        model_regularization: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        optimizer_regularization: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        average_probabilities: dict = None,
        current_epoch: int = None,
        current_fl_round: int = None,
        max_num_epochs: int = None,
        node_id: int = 0,
        wandb_run=None,
        sigma_update_lambda: float = None,
        epoch: int = 0,
    ) -> dict:
        """This function is used to train the private model.
        If we want to use the unfairness mitigation through Regularization
        and Differential Privacy then we need two models: one for the
        classic training from which we get the loss and one for the
        computation of the regularization term.
        The reason why we need two models is that we can't directly sum the two losses
        when using DP with Opacus because of how the library is implemented.
        For more details about this, refer to this open issue in the Opacus repository:
        https://github.com/pytorch/opacus/issues/249.
        The only working solution that we have found to overcome this issue and to
        inject the regularization term into the training is to
        sum the per sample gradients of the two models.

        Args:
            train_parameters (RegularizationConfig): the configuration of all the possible
                settings to train the model
            model (torch.nn.Module): the model to train
            model_regularization (torch.nn.Module): the model used to compute
                the regularization term. None if we do not want to use
                the regularization and DP
            optimizer (torch.optim.Optimizer): the optimizer used to train the
                model
            optimizer_regularization (torch.optim.Optimizer): the optimizer used
                to train the model_regularization if we want to use
                the regularization and DP
            node_id (int): id of the node. this is used only in FL
            train_loader (torch.utils.data.DataLoader): the training set
            test_loader (torch.utils.data.DataLoader): the test set
            current_epoch (int): the current epoch
            average_probabilities (dict): the average probabilities computed by the server
                these are needed when using Federated Learning. The default value is None
            sigma_update_lambda (float): the sigma parameter used to update the lambda
                ensuring Differential Privacy
            epoch (int): the current epoch
            current_fl_round (int): the current fl round in which the client was selected
            max_num_epochs (int): the maximum number of epochs to train the model
            wandb_run (wandb.Run): the wandb run used to log the metrics

        Raises:
            ValueError: if model_regularization is None and DPL is True
        """

        criterion = nn.CrossEntropyLoss()
        if train_parameters.metric == "disparity":
            criterion_regularization = RegularizationLoss()
        else:
            raise ValueError("The metric is not supported")
        losses = []
        losses_with_regularization = []
        total_correct = 0
        total = 0
        velocity = 0
        history_lambda = []

        alpha = None
        model.train()
        if model_regularization:
            model_regularization.train()

        is_tunable = (
            True if train_parameters.regularization_mode == "tunable" else False
        )

        regularization_term = None

        model.train()

        fairness_violation = None
        MAX_PHYSICAL_BATCH_SIZE = 512
        with BatchMemoryManager(
            data_loader=train_loader,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
            optimizer=optimizer,
        ) as memory_safe_data_loader:
            for batch_number, (data, sensitive_feature, target) in enumerate(
                memory_safe_data_loader, 0
            ):
                regularization_term = None
                # If we use DPL, we need to synchronize the two models
                # before each batch because during the training we only update
                # the model and not the model_regularization. Therefore, we need
                # to be sure to have the same weights in the two models.
                # The model_regularization is only used to compute the regularization term
                # and the per sample gradient wrt the regularization term.
                if model_regularization is not None:
                    Utils.sync_models(model_regularization, model)

                optimizer.zero_grad()
                target = target.long()
                target = target.to(train_parameters.device)
                data = data.to(train_parameters.device)
                sensitive_feature = sensitive_feature.to(train_parameters.device)

                if train_parameters.regularization and not model_regularization:
                    raise ValueError(
                        "model_regularization can't be None if DPL and private are True"
                    )

                # If DPL is True, we need to compute the regularization term
                # and add it to the loss. The computation of the regularization
                # term is done as in the classic training, but with the model
                # that is used to compute the regularization term.
                if train_parameters.regularization and model_regularization:
                    # We first get the output of the model_regularization
                    output_regularization = model_regularization(data)
                    fairness_violation = None
                    # And then we use these outputs in the computation of the regularization term
                    fairness_violation = Learning.compute_regularization_term(
                        data=data,
                        targets=target,
                        sensitive_feature=sensitive_feature,
                        train_parameters=train_parameters,
                        criterion_regularization=criterion_regularization,
                        outputs=output_regularization,
                        average_probabilities=average_probabilities,
                        batch=(epoch + 1) * batch_number,
                    )

                    if fairness_violation is not None and fairness_violation > 0:
                        # Here the fairness_violation is multiplied with the lambda
                        # and then we compute the backward pass on it.
                        # We do the multiplication with the lambda because we want to
                        # give a certain weight lambda to the unfairness
                        # regularization term and a weight (1 - lambda) to the classic loss
                        regularization_term = (
                            train_parameters.regularization_lambda * fairness_violation
                        )

                        try:
                            regularization_term.backward()
                        except Exception:
                            print(
                                "EXCEPTION while computing the backward pass: Node id ",
                                node_id,
                                "Outpus: ",
                                len(data),
                                "target: ",
                                len(target),
                                "sensitive_attribute_list: ",
                                len(sensitive_feature),
                                "FAIRNESS VIOLATION: ",
                                fairness_violation,
                                "reg term: ",
                                regularization_term,
                                "current sens features: ",
                                set([item.item() for item in sensitive_feature]),
                            )

                if wandb_run:
                    if regularization_term:
                        wandb_run.log(
                            {
                                "batch": (epoch + 1) * batch_number,
                                "Unfairness metric Batch": (
                                    regularization_term.item()
                                    if isinstance(regularization_term, torch.Tensor)
                                    else regularization_term
                                ),
                            }
                        )
                    else:
                        wandb_run.log(
                            {
                                "batch": (epoch + 1) * batch_number,
                                "Unfairness metric Batch": 0,
                            }
                        )

                # Now we can compute the output of the model and the classic loss
                outputs = model(data)
                history_lambda.append(train_parameters.regularization_lambda)
                classic_loss = criterion(outputs, target)

                # The classic loss is multiplied with (1 - lambda) because we want to
                # weight this loss with (1 - lambda) and the regularization term with lambda
                # then we compute the backward pass on it.
                if regularization_term is not None:
                    loss = (1 - train_parameters.regularization_lambda) * classic_loss
                else:
                    loss = classic_loss

                # We store the computed losses for logging purposes
                losses.append(loss.item())
                if regularization_term is not None:
                    losses_with_regularization.append(
                        (regularization_term + loss).item()
                    )

                # And we do the backward pass to compute the per sample gradients
                # (when we use DP)
                loss.backward()
                # If we use regularization, we need to sum the per sample gradients of the
                # two models so that we can update the model considering both
                # the classic loss and the regularization term
                if regularization_term and train_parameters.regularization_lambda > 0:
                    for p1, p2 in zip(
                        model.parameters(), model_regularization.parameters()
                    ):
                        if p1.grad_sample is not None and p2.grad_sample is not None:
                            p1.grad_sample += p2.grad_sample

                # With this call we update the model using the optimizer
                # Note that the model that we update is "model" and not "model_regularization"
                # This is why we then need to synchronize the two models
                optimizer.step()

                optimizer.zero_grad()
                if optimizer_regularization:
                    optimizer_regularization.zero_grad()

                # Compute the total number of correct predictions
                # and store the total number of predictions
                # We will need these information to compute the
                # accuracy of the model and to compute the maximum disparity
                # of the model after the end of this epochPtabu
                _, predicted = torch.max(outputs.data, 1)
                correct = (predicted == target).float().sum()
                total_correct += correct
                total += target.size(0)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Alpha is like a learning rate but it is used to update the lambda
                # it tells us how much we want to be aggressive when updating the lambda
                alpha = train_parameters.alpha
                # When we have a target and we are in a tunable scenario then we have to
                # update the lamdba.
                if (
                    fairness_violation is not None
                    and train_parameters.target
                    and is_tunable
                ):
                    model.eval()
                    output_regularization = model(data)
                    fairness_violation = None
                    fairness_violation = Learning.compute_regularization_term(
                        data=data,
                        targets=target,
                        sensitive_feature=sensitive_feature,
                        train_parameters=train_parameters,
                        criterion_regularization=criterion_regularization,
                        outputs=output_regularization,
                        average_probabilities=average_probabilities,
                        batch=(epoch + 1) * batch_number,
                    )
                    model.train()

                    # When we use DP we need to protect the update of the Lambda
                    # adding noise to the fairness_violation. We need to do this because
                    # the lambda depends on the fairness violation that is a metric
                    # that is computed using the entire batch of data. We're not doing
                    # a "per-sample" unfairness computation and so the update of the
                    # lambda is not protected by the DP-SGD.
                    if sigma_update_lambda:
                        # We call our get_noise function method to get the noise
                        # that we need to add to the fairness_violation
                        noise = Utils.get_noise(
                            mechanism_type="gaussian", sigma=sigma_update_lambda
                        )
                    else:
                        noise = 0

                    # The noise is summed to the fairness violation of the current batch
                    # that is the regularization term that we used in this batch
                    fairness_violation_with_noise = fairness_violation.item() + noise

                    # We compute the distance between the target and the actual
                    # disparity of the trained model
                    delta = train_parameters.target - (fairness_violation_with_noise)

                    # We use the implementation of the Momentum as in Pytorch
                    # to have a smoother update of the lambda
                    # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
                    velocity = train_parameters.momentum * velocity + delta
                    # The alpha is an hyperparameter and it tells us how much we want to
                    # be fast when updating the lambda
                    new_lambda = (
                        train_parameters.regularization_lambda - alpha * velocity
                    )

                    # We need to clip the lambda so that it is in the range (0,1).
                    # This does not affect the DP because of post-processing
                    # guarantees of DP-SGD
                    if new_lambda >= 0 and new_lambda <= 1:
                        train_parameters.regularization_lambda = new_lambda
                    elif new_lambda > 1:
                        train_parameters.regularization_lambda = 1
                    else:
                        train_parameters.regularization_lambda = 0

        # If we have specified it, then we have to update the alpha using the weight decay
        if train_parameters.weight_decay_alpha:
            alpha = exp_lr_scheduler(
                initial_alpha=alpha,
                current_fl_round=current_fl_round,
                decay_rate=train_parameters.weight_decay_alpha,
            )
        train_parameters.alpha = alpha
        train_loss = np.mean(losses)
        train_loss_with_regularization = np.mean(losses_with_regularization)

        accuracy = total_correct / total

        if train_parameters.metric == "disparity":
            # We compute the fairness metric on the entire train dataset
            # at the end of the training
            max_unfairness_train = criterion_regularization.violation_with_dataset(
                model=model,
                dataset=train_loader,
                device=train_parameters.device,
                average_probabilities=average_probabilities,
            )
        else:
            raise ValueError("The metric is not supported")

        return {
            "epoch": current_epoch,
            "Train Loss": train_loss,
            "Train Loss + Regularizaion": train_loss_with_regularization,
            "Train Accuracy": accuracy,
            "Max Unfairness Train": max_unfairness_train,
            "history_lambda": history_lambda,
        }

    @staticmethod
    def test(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        train_parameters: RegularizationConfig,
        current_epoch: int,
        set_name: str = "Test set",
        average_probabilities=None,
    ) -> Tuple[float, float, float, float, float, float]:
        """Test the model on the test set computing the
        accuracy and also the maximum disparity of the model.

        Args:
        Args:
            model (torch.nn.Module): The model we want to test
            test_loader (torch.utils.data.DataLoader): the test dataset
            train_parameters (RegularizationConfig): the parameters used for the training
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
                target = target.long()
                data, target = (
                    data.to(train_parameters.device),
                    target.to(train_parameters.device),
                )
                output = model(data)
                total += target.size(0)
                test_loss = criterion(output, target)
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

        if train_parameters.metric == "disparity":
            criterion_regularization = RegularizationLoss()
            # We compute the violation on the entire test set with the current model
            unfairness_test = criterion_regularization.violation_with_dataset(
                model=model,
                dataset=test_loader,
                device=train_parameters.device,
                average_probabilities=average_probabilities,
            )

        # if losses is on gpu we need to move it back to cpu
        losses = [item.item() for item in losses]
        test_loss = np.mean(losses)
        accuracy = correct / total

        y_true = [item.item() for item in y_true]
        y_pred = [item.item() for item in y_pred]

        f1score = f1_score(y_true, y_pred, average="macro")
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")

        return (
            test_loss,
            accuracy,
            f1score,
            precision,
            recall,
            unfairness_test,
            y_true,
            y_pred,
            colors,
        )

    @staticmethod
    def compute_regularization_term(
        data: torch.utils.data.DataLoader,
        targets: torch.tensor,
        sensitive_feature: torch.tensor,
        criterion_regularization: RegularizationLoss,
        train_parameters: RegularizationConfig,
        outputs: torch.tensor,
        average_probabilities: dict,
        wandb_run=None,
        batch: int = 0,
    ) -> torch.tensor:
        """This function computes the regularization term on the training data
        passed as parameter.

        Args:
            data (torch.utils.data.DataLoader): the dataset on which the regularization
                term is computed
            target (torch.tensor): the targets of the data we pass as parameter
            sensitive_feature (torch.tensor): the corresponding sensitive features
            criterion_regularization (RegularizationLoss): the regularization criterion
            train_parameters (RegularizationConfig): the parameters used for the training
            outputs (torch.tensor): Output of the model
            average_probabilities (dict): the probabilities computed by the server. This is only
                used when using FL

        Returns:
            fairness_violation (_type_): the fairness violation computed on the data
                This does not include the multiplication with the Lambda
                and the Backward pass.
        """
        possible_targets = set([item.item() for item in targets])
        possible_sensitive_attributes = set([item.item() for item in sensitive_feature])

        if train_parameters.metric == "disparity":
            fairness_violation = criterion_regularization(
                sensitive_attribute_list=sensitive_feature,
                device=train_parameters.device,
                predictions=outputs,
                possible_sensitive_attributes=list(possible_sensitive_attributes),
                possible_targets=list(possible_targets),
                average_probabilities=average_probabilities,
                wandb_run=wandb_run,
                batch=batch,
            )
        else:
            raise ValueError("The metric is not supported")

        return fairness_violation

    @staticmethod
    def test_prediction(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        train_parameters: RegularizationConfig,
        current_epoch: int,
        set_name: str = "test set",
    ) -> Tuple[float, float, float, float, float, float]:
        """Test the model on the test set computing the
        accuracy and also the maximum disparity of the model.

        Args:
            model (torch.nn.Module): The model we want to test
            test_loader (torch.utils.data.DataLoader): the test dataset
            train_parameters (RegularizationConfig): the parameters used for the training
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
                target = target.long()
                data, target = (
                    data.to(train_parameters.device),
                    target.to(train_parameters.device),
                )
                output = model(data)
                total += target.size(0)
                # if train_parameters.tabular_data:
                test_loss = criterion(output, target)
                # else:
                # test_loss = criterion(output, target)
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
            y_true,
        )
