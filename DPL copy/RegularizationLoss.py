from collections import Counter
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class RegularizationLoss(nn.Module):
    """This class defines the regularization loss as proposed in
    https://arxiv.org/abs/2302.09183.
    """

    def __init__(self, weight=None, size_average=True, estimation=0.5) -> None:
        """Initialization of the regularization loss."""
        super(RegularizationLoss, self).__init__()
        self.estimation = estimation

    def forward(
        self,
        sensitive_attribute_list: torch.tensor,
        device: torch.device,
        predictions: torch.tensor,
        possible_sensitive_attributes: list,
        possible_targets: list,
        wandb_run=None,
        average_probabilities=None,
        train_parameters: dict = None,
        test: str = "FALSE",
    ) -> torch.tensor:
        """This function computes the regularization term.
        It takes as input the sensitive attribute list, the targets,
        the device and the predictions
        compute with the model. It returns the regularization term.

        What we do here:
        - We compute the softmax of the predictions
        - Then we consider the possible combinations of targets and sensitive features
            and we compute the corresponding fairness violation term
        - We return the maximum violation term among all the possible combinations

        Args:
            sensitive_attribute_list (torch.Tensor): a list with the value of
                the sensitive attribute for each sample in the batch
            device (_type_): the device we're using to train the model
            predictions (_type_): the output of the model for the current batch
            sensitive_attributes (_type_): the possible values of the sensitive
                attribute
            possible_targets (_type_): the possible target values we have in this
                dataset

        Example:
            >>> sensitive_attribute_list = torch.tensor([1, 1, -1, -1, 1, -1])
            >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            >>> predictions = torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7],
                [0.4, 0.6], [0.5, 0.5], [0.6, 0.4]])
            >>> possible_sensitive_attributes = [1, -1]
            >>> possible_targets = [0, 1]
            >>> regularization_loss = RegularizationLoss()
            >>> regularization_loss(sensitive_attribute_list, device, predictions,
                possible_sensitive_attributes, possible_targets)

        Returns:
            float: the disparity metric computed on the data passed as parameter
        """
        fairness_violations = []

        # We compute the softmax of the predictions. We do this because
        # we can't use the argmax function on the nn output,
        # because we need differentiable results
        softmax_ = F.softmax(predictions, dim=1)

        # We compute the argmax of the predictions, this is used to count
        # the number of samples for each class that are predicted with one class
        # or with the other.
        predictions_argmax = torch.argmax(torch.tensor(predictions), dim=1).to(device)
        for target in possible_targets:
            for z in possible_sensitive_attributes:
                # Z_eq_z and Z_not_eq_z are the denominators that we will use
                # in the DPL formula. |Z=z| and |Z!=z|
                Z_eq_z = len(sensitive_attribute_list[sensitive_attribute_list == z])

                Z_not_eq_z = len(
                    sensitive_attribute_list[sensitive_attribute_list != z]
                )

                # We get the number of samples that are predicted with the target class
                # target and that have the sensitive attribute equal
                #  to z:  |Y = k, Z = z|.
                # In this case we just sum the columns of the rows that
                # respect the previous constraint.
                # Example: Given [[0.2, 0.8], [0.4, 0.6], [0.3, 0.7]], suppose
                # that to compute Y_eq_k_and_Z_eq_z we have to consider only
                # the first and the third row and that we are considering the class 1.
                # In this case we will sum 0.8 and 0.7.
                Y_eq_k_and_Z_eq_z = torch.sum(
                    softmax_[
                        (predictions_argmax == target) & (sensitive_attribute_list == z)
                    ][:, target]
                )

                # Here we compute |Y = k, Z != z| with the same strategy we used to
                # compute |Y = k, Z = z|.
                Y_eq_k_and_Z_not_eq_z = torch.sum(
                    softmax_[
                        (predictions_argmax == target) & (sensitive_attribute_list != z)
                    ][:, target]
                )

                # Now we can compute the violation term that we will
                # sum to our loss. We have to consider the case in which
                # Z_eq_z or Z_not_eq_z are equal to 0, because in this case
                # we will have a division by 0.
                # If this doesn't happen we can compute the violation term
                # using the formula of the paper.
                # |P(Y=y|Z=z) - P(Y=y|Z!=z)|
                print(
                    f"{test}: Target {target}, sensitive value: {z}, {Counter([item.item() for item in predictions_argmax])}  TOTALE SAMPLES: {len(softmax_)} Y_eq_k_and_Z_eq_z {Y_eq_k_and_Z_eq_z}, Z_eq_z {Z_eq_z}, Y_eq_k_and_Z_not_eq_z {Y_eq_k_and_Z_not_eq_z}, Z_not_eq_z {Z_not_eq_z}"
                )

                if Z_eq_z == 0 and Z_not_eq_z != 0:
                    denominator = 1 if z == 1 or z == 1.0 else 0
                    if (
                        train_parameters
                        and train_parameters.probability_estimation
                        and average_probabilities
                        and average_probabilities.get(f"{target}|{denominator}", None)
                    ):
                        violation_term = torch.abs(
                            average_probabilities[f"{target}|{denominator}"]
                            - 
                            Y_eq_k_and_Z_not_eq_z / Z_not_eq_z
                        )
                        value = average_probabilities[f"{target}|{denominator}"]
                        print(f"Using {value} as estimation - the violation term is {violation_term} - The real value is {Y_eq_k_and_Z_not_eq_z / Z_not_eq_z}")
                    else:
                        violation_term = torch.abs(
                            Y_eq_k_and_Z_not_eq_z / Z_not_eq_z
                        ) - torch.abs(Y_eq_k_and_Z_not_eq_z / Z_not_eq_z)
                elif Z_not_eq_z == 0 and Z_eq_z != 0:
                    denominator = 1 if z == 0 else 0
                    if (
                        train_parameters
                        and train_parameters.probability_estimation
                        and average_probabilities
                        and average_probabilities.get(f"{target}|{denominator}", None)
                    ):
                        violation_term = torch.abs(
                            (Y_eq_k_and_Z_eq_z / Z_eq_z)
                            - average_probabilities[f"{target}|{denominator}"]
                        )
                        value = average_probabilities[f"{target}|{denominator}"]
                        print(f"Using {value} as estimation - the violation term is {violation_term} - The real value is {(Y_eq_k_and_Z_eq_z / Z_eq_z)}")
                    else:
                        violation_term = torch.abs(
                            Y_eq_k_and_Z_eq_z / Z_eq_z
                        ) - torch.abs(Y_eq_k_and_Z_eq_z / Z_eq_z)


                # controllare se ho fatto che i nodi che hanno meno dati non hanno tutto il maschio/femmina o se invece
                # non hanno solamente la combinazione smiling/gender.
                else:
                    denominator = 1 if z == 0 else 0
                    if (
                        train_parameters
                        and train_parameters.probability_estimation
                        and average_probabilities
                        and average_probabilities.get(f"{target}|{denominator}", None)
                    ):
                        violation_term = torch.abs(
                            (Y_eq_k_and_Z_eq_z / Z_eq_z)
                            - average_probabilities[f"{target}|{denominator}"]
                        )

                    else:
                        violation_term = torch.abs(
                            (Y_eq_k_and_Z_eq_z / Z_eq_z)
                            - (Y_eq_k_and_Z_not_eq_z / Z_not_eq_z)
                        )


                violation_with_argmax = self.compute_violation_with_argmax(
                    predictions_argmax=predictions_argmax,
                    sensitive_attribute_list=sensitive_attribute_list,
                    current_target=target,
                    current_sensitive_feature=z,
                )
                if wandb_run:
                    wandb_run.log(
                        {
                            f"violation_with_argmax_Y={target}_Z={z}": violation_with_argmax,
                            f"violation_Y={target}_Z={z}": violation_term.item()
                            if isinstance(violation_term, torch.Tensor)
                            else violation_term,
                            f"Prob(Y_eq_{target}_and_Z_eq_{z})": (
                                Y_eq_k_and_Z_eq_z / Z_eq_z
                            )
                            if Z_eq_z != 0
                            else 0,
                        }
                    )

                fairness_violations.append(violation_term)

        fairness_violations_ = [
            item.item() if isinstance(item, torch.Tensor) else item
            for item in fairness_violations
        ]
        index = fairness_violations_.index(max(fairness_violations_))

        fairness_violations = torch.stack(fairness_violations)

        mask = torch.full((fairness_violations.shape[0],), 0, dtype=torch.float32).to(
            device
        )
        mask[index] = 1

        res = torch.sum(mask * fairness_violations)



        return res

    def violation_with_dataset(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.DataLoader,
        device: torch.device,
        test: str = "FALSE",
        average_probabilities: dict = None, 
        train_parameters: dict = None,
    ) -> torch.tensor:
        """
        When we want to compute the disparity metric on the entire dataset
        we can't directly use the forward function because we don't have the
        predictions and the sensitive attribute list for each batch.
        So in this function we just use the model to compute the predictions for
        all the samples in the dataset and we aggregate the results in a single
        final tensor that we pass to the forward function.
        This is used, for instance, to compute the disparity of the model
        on the test dataset

        Args:
            model (torch.nn.Module): the model we want to evaluate
            dataset (torch.utils.data.DataLoader): the dataset we want to
                use during the evaluation
            device (torch.device): the device we're using to train the model

        Returns:
            float: the disparity metric computed on the dataset
                passed as parameter
        """
        predictions = torch.tensor([]).to(device)
        sensitive_attribute_list = torch.tensor([]).to(device)
        targets = []
        model.eval()
        with torch.no_grad():
            for images, sensitive_attributes, target in dataset:
                images = images.to(device)
                target = target.to(device)

                output = model(images)

                predictions = torch.cat((predictions, output), 0)
                sensitive_attribute_list = torch.cat(
                    (sensitive_attribute_list, sensitive_attributes.to(device)), 0
                )
                targets += target.tolist()

        sensitive_attributes = list(
            set([item.item() for item in sensitive_attribute_list])
        )
        target_list = list(set(targets))

        return self.forward(
            sensitive_attribute_list,
            device,
            predictions,
            sensitive_attributes,
            target_list,
            test=test,
            average_probabilities=average_probabilities,
            train_parameters=train_parameters,
        )

    def compute_violation_with_argmax(
        self,
        predictions_argmax: torch.tensor,
        sensitive_attribute_list: torch.tensor,
        current_target: int,
        current_sensitive_feature: int,
    ) -> Tuple[int, int, int, int]:
        """Debug function used to compute the DPL using the argmax function
        instead of the softmax.

        Args:
            predictions_argmax (torch.tensor): predictions of the model
            sensitive_attribute_list (torch.tensor): _description_
            target (int): The target we are considering
                in this iteration to compute the violation
            sensitive_feature (int): the sensitive feature
                we are considering in this iteration to
                compute the violation

        Returns:
            Tuple[int, int, int, int]: The number of times the
                prediction is equal to the target and the sensitive
                feature is equal to the sensitive feature we are
                considering in this iteration, the number of times
                the sensitive feature is equal to the sensitive
                feature we are considering in this iteration, the
                number of times the prediction is equal to the target
                and the sensitive feature is not equal to the sensitive
                feature we are considering in this iteration, the number
                of times the sensitive feature is not equal to the sensitive
                feature we are considering in this iteration
        """
        counter_sensitive_features = Counter(
            [item.item() for item in sensitive_attribute_list]
        )

        Z_eq_z_argmax = counter_sensitive_features[current_sensitive_feature]
        Z_not_eq_z_argmax = counter_sensitive_features[-current_sensitive_feature]
        Y_eq_k_and_Z_eq_z_argmax = 0
        Y_eq_k_and_Z_not_eq_z_argmax = 0

        for prediction, sensitive_feature in zip(
            predictions_argmax, sensitive_attribute_list
        ):
            if (
                prediction == current_target
                and sensitive_feature == current_sensitive_feature
            ):
                Y_eq_k_and_Z_eq_z_argmax += 1
            elif prediction == current_target and sensitive_feature == -(
                current_sensitive_feature
            ):
                Y_eq_k_and_Z_not_eq_z_argmax += 1

        if Z_eq_z_argmax == 0 and Z_not_eq_z_argmax != 0:
            return np.abs(Y_eq_k_and_Z_not_eq_z_argmax / Z_not_eq_z_argmax).item()
        elif Z_eq_z_argmax != 0 and Z_not_eq_z_argmax == 0:
            return np.abs(Y_eq_k_and_Z_eq_z_argmax / Z_eq_z_argmax).item()
        else:
            np.abs(
                Y_eq_k_and_Z_eq_z_argmax / Z_eq_z_argmax
                - Y_eq_k_and_Z_not_eq_z_argmax / Z_not_eq_z_argmax
            ).item()

    @staticmethod
    def compute_probabilities(
        predictions,
        sensitive_attribute_list,
        device: torch.device,
        possible_sensitive_attributes: list,
        possible_targets: list,
    ) -> torch.tensor:
        fairness_violations = []

        softmax_ = F.softmax(predictions, dim=1)

        # We compute the argmax of the predictions, this is used to count
        # the number of samples for each class that are predicted with one class
        # or with the other.
        predictions_argmax = torch.argmax(torch.tensor(predictions), dim=1).to(device)
        sensitive_attribute_list = sensitive_attribute_list.to(device)
        probabilities = {}
        for target in list(possible_targets):
            for z in possible_sensitive_attributes:
                # Z_eq_z and Z_not_eq_z are the denominators that we will use
                # in the DPL formula. |Z=z| and |Z!=z|
                Z_eq_z = len(sensitive_attribute_list[sensitive_attribute_list == z])

                # Y_eq_k_and_Z_eq_z = len(
                #     predictions_argmax[
                #         (predictions_argmax == target) & (sensitive_attribute_list == z)
                #     ]
                # )

                Y_eq_k_and_Z_eq_z = torch.sum(
                    softmax_[
                        (predictions_argmax == target) & (sensitive_attribute_list == z)
                    ][:, target]
                )

                probabilities[f"{target}|{z}"] = Y_eq_k_and_Z_eq_z
                probabilities[f"{z}"] = Z_eq_z

        print(f"----> NODE COMPUTED {probabilities}")

        return probabilities
