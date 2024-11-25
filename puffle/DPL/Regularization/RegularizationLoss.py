import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class RegularizationLoss(nn.Module):
    """This class defines the regularization loss as proposed in
    https://arxiv.org/abs/2302.09183.
    It uses the definition of demographic parity to compute the
    fairness violation term for each batch and then it uses this
    violation term as a regularization term to add to the loss.
    """

    def __init__(self, weight=None, size_average=True, estimation=0.5) -> None:
        """Initialization of the regularization loss."""
        super().__init__()
        self.estimation = estimation

    def forward(
        self,
        sensitive_attribute_list: torch.tensor,
        device: torch.device,
        predictions: torch.tensor,
        possible_sensitive_attributes: list,
        possible_targets: list,
        average_probabilities: dict = None,
        wandb_run=None,
        batch=None,
    ) -> torch.tensor:
        """This function computes the regularization term.
        It takes as input the sensitive attribute list, the targets,
        the device and the predictions computed with the model.
        It returns the regularization term.

        What we do here:
        - We compute the softmax of the predictions. We do this to have the output of
           this function in the same scale of the loss of the model. Moreover, we cannot
           use the argmax function on the nn output, because we need differentiable results.
        - Then we consider the possible combinations of targets and sensitive features
            and we compute the corresponding fairness violation term. In this case
            the metric we use is the Demographic Parity Loss.
        - We return the maximum violation term among all the possible combinations

        Args:
            sensitive_attribute_list (np.array): a list with the value of
                the sensitive attribute for each sample in the batch
            device (str): the device we're using to train the model (gpu or cpu)
            predictions (np.array): the predictions of the model for the batch of data
            possible_sensitive_attributes (list): the possible values of the sensitive
                attribute
            possible_targets (list): the possible target values we have in this
                dataset
            average_probabilities (dict): in case of Federated learning, if a client
                has only a subset of the possible sensitive attributes, we can use the
                average probabilities of the other clients to estimate the probabilities
                of the missing sensitive attributes. This is None in centralised learning
            wandb_run (wandb.Run): the wandb run we're using to log the metrics
            batch (int): the number of the batch we're considering

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

        # convert the list of sensitive attributes to a tensor and move it to the device
        sensitive_attribute_list = torch.tensor(
            [int(item) for item in sensitive_attribute_list]
        )
        sensitive_attribute_list = sensitive_attribute_list.to(device)

        # We compute the argmax of the predictions, this is used to count
        # the number of samples for each class that are predicted with one class
        # or with the other.
        predictions_argmax = torch.argmax(torch.tensor(predictions), dim=1).to(device)
        # we convert the possible targets and the possible sensitive attributes to a list
        # just to be sure that the values are integers
        possible_targets = [int(item) for item in possible_targets]
        possible_sensitive_attributes = [
            int(item) for item in possible_sensitive_attributes
        ]

        for target in possible_targets:
            for z in possible_sensitive_attributes:
                # Z_eq_z and Z_not_eq_z are the denominators that we will use
                # in the DPL formula. |Z=z| and |Z!=z|
                Z_eq_z = len(sensitive_attribute_list[sensitive_attribute_list == z])
                Z_not_eq_z = len(
                    sensitive_attribute_list[sensitive_attribute_list != z]
                )

                # We get the number of samples that are predicted with the target class
                # target and that have the sensitive attribute equal to z:  |Y = k, Z = z|.
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
                # using the classic formula |P(Y=y|Z=z) - P(Y=y|Z!=z)|
                # If this happens, instead, we have to use the estimation

                if (Y_eq_k_and_Z_eq_z == 0 and Y_eq_k_and_Z_not_eq_z != 0) or (
                    Z_eq_z == 0 and Z_not_eq_z != 0
                ):
                    denominator = 1 if z == 1 else 0
                    if average_probabilities and average_probabilities.get(
                        f"{target}|{denominator}", None
                    ):
                        # In this case I use the estimation
                        violation_term = torch.abs(
                            average_probabilities[f"{target}|{denominator}"]
                            - Y_eq_k_and_Z_not_eq_z / Z_not_eq_z
                        )
                    else:
                        # In this case instead I return 0, this only happens when
                        # we do not have the estimation (for instance in the first
                        # FL Round)
                        violation_term = torch.abs(
                            Y_eq_k_and_Z_not_eq_z / Z_not_eq_z
                        ) - torch.abs(Y_eq_k_and_Z_not_eq_z / Z_not_eq_z)
                elif (Y_eq_k_and_Z_eq_z != 0 and Y_eq_k_and_Z_not_eq_z == 0) or (
                    Z_not_eq_z == 0 and Z_eq_z != 0
                ):
                    # This is just the other case
                    denominator = 1 if z == 0 else 0
                    if average_probabilities and average_probabilities.get(
                        f"{target}|{denominator}", None
                    ):
                        violation_term = torch.abs(
                            (Y_eq_k_and_Z_eq_z / Z_eq_z)
                            - average_probabilities[f"{target}|{denominator}"]
                        )
                    else:
                        violation_term = torch.abs(
                            Y_eq_k_and_Z_eq_z / Z_eq_z
                        ) - torch.abs(Y_eq_k_and_Z_eq_z / Z_eq_z)

                else:
                    # In this case we have all the combinations,
                    # so we can compute the violation term using the classic formula
                    violation_term = torch.abs(
                        (Y_eq_k_and_Z_eq_z / Z_eq_z)
                        - (Y_eq_k_and_Z_not_eq_z / Z_not_eq_z)
                    )

                fairness_violations.append(violation_term)

        fairness_violations_ = [
            item.item() if isinstance(item, torch.Tensor) else item
            for item in fairness_violations
        ]

        # We get the index of the maximum violation term. Then we create a mask with
        # all zeros and we set to 1 the element at the index we found. We use this mask
        # to sum the violation terms and we return the result. This was needed because
        # when we started to work on this project we discovered that without this
        # some of the gradients were not computed correctly. I would not remove it
        # even if I'm not sure that it is needed anymore.
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
        average_probabilities: dict,
        device: torch.device,
    ) -> torch.tensor:
        """
        When we want to compute the disparity metric on the entire dataset
        we can't directly use the forward function because we don't have the
        predictions and the sensitive attribute list for each batch.
        So in this function we just use the model to compute the predictions for
        all the samples in the dataset and we aggregate the results in a single
        final tensor that we pass to the forward function.
        This is used, for instance, to compute the disparity of the model
        on the test dataset.

        Args:
            model (torch.nn.Module): the model we want to evaluate
            dataset (torch.utils.data.DataLoader): the dataset we want to
                use during the evaluation
            average_probabilities (dict): in case of Federated learning, if a client
                has only a subset of the possible sensitive attributes, we can use the
                average probabilities of the other clients to estimate the probabilities
                of the missing sensitive attributes. This is None in centralised learning
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
            for images, sensitive_attributes, _, target in dataset:
                images = images.to(device)
                target = target.to(device)

                output = model(images)

                predictions = torch.cat((predictions, output), 0)
                sensitive_attribute_list = torch.cat(
                    (sensitive_attribute_list, sensitive_attributes.to(device)), 0
                )
                targets += target.tolist()

        sensitive_attributes = list({item.item() for item in sensitive_attribute_list})
        target_list = list(set(targets))

        # now we just call the forward function with the "fake" predictions and the sensitive
        # attribute list we computed
        return self.forward(
            sensitive_attribute_list,
            device,
            predictions,
            sensitive_attributes,
            target_list,
            average_probabilities=average_probabilities,
        )

    def violation_with_dataset_2(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.DataLoader,
        average_probabilities: dict,
        device: torch.device,
    ) -> torch.tensor:
        """
        When we want to compute the disparity metric on the entire dataset
        we can't directly use the forward function because we don't have the
        predictions and the sensitive attribute list for each batch.
        So in this function we just use the model to compute the predictions for
        all the samples in the dataset and we aggregate the results in a single
        final tensor that we pass to the forward function.
        This is used, for instance, to compute the disparity of the model
        on the test dataset.

        Args:
            model (torch.nn.Module): the model we want to evaluate
            dataset (torch.utils.data.DataLoader): the dataset we want to
                use during the evaluation
            average_probabilities (dict): in case of Federated learning, if a client
                has only a subset of the possible sensitive attributes, we can use the
                average probabilities of the other clients to estimate the probabilities
                of the missing sensitive attributes. This is None in centralised learning
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
            for images, _, sensitive_attributes, target in dataset:
                images = images.to(device)
                target = target.to(device)

                output = model(images)

                predictions = torch.cat((predictions, output), 0)
                sensitive_attribute_list = torch.cat(
                    (sensitive_attribute_list, sensitive_attributes.to(device)), 0
                )
                targets += target.tolist()

        sensitive_attributes = list({item.item() for item in sensitive_attribute_list})
        target_list = list(set(targets))

        # now we just call the forward function with the "fake" predictions and the sensitive
        # attribute list we computed
        return self.forward(
            sensitive_attribute_list,
            device,
            predictions,
            sensitive_attributes,
            target_list,
            average_probabilities=average_probabilities,
        )

    def compute_violation_with_argmax(
        self,
        predictions_argmax: torch.tensor,
        sensitive_attribute_list: torch.tensor,
        current_target: int,
        current_sensitive_feature: int,
        weights: dict = None,
    ):
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

        opposite_sensitive_feature = 0 if current_sensitive_feature == 1 else 1

        Z_eq_z_argmax = 0
        Z_not_eq_z_argmax = 0
        Y_eq_k_and_Z_eq_z_argmax = 0
        Y_eq_k_and_Z_not_eq_z_argmax = 0

        for prediction, sensitive_feature in zip(
            predictions_argmax, sensitive_attribute_list
        ):
            current_weight = (
                weights.get(f"(Y={int(prediction)}, Z={int(sensitive_feature)})", 1)
                if weights
                else 1
            )

            if sensitive_feature == current_sensitive_feature:
                Z_eq_z_argmax += 1 * current_weight
            else:
                Z_not_eq_z_argmax += 1 * current_weight

            if (
                prediction == current_target
                and sensitive_feature == current_sensitive_feature
            ):
                Y_eq_k_and_Z_eq_z_argmax += 1 * current_weight
            elif (
                prediction == current_target
                and sensitive_feature == opposite_sensitive_feature
            ):
                Y_eq_k_and_Z_not_eq_z_argmax += 1 * current_weight

        if Z_eq_z_argmax == 0 and Z_not_eq_z_argmax != 0:
            return np.abs(Y_eq_k_and_Z_not_eq_z_argmax / Z_not_eq_z_argmax).item()
        elif Z_eq_z_argmax != 0 and Z_not_eq_z_argmax == 0:
            return np.abs(Y_eq_k_and_Z_eq_z_argmax / Z_eq_z_argmax).item()
        elif Z_eq_z_argmax == 0 and Z_not_eq_z_argmax == 0:
            return 0
        else:
            return np.abs(
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
        """This function computes the probabilities and the counters
            of each possible combination of target and sensitive attribute.
            It is used to compute the probabilities that we use to estimate
            the probabilities of the missing sensitive attributes in the
            Federated Learning scenario.

        Args:
            sensitive_attribute_list: a list with the value of
                the sensitive attribute for each sample in the batch
            device: the device we're using to train the model
            possible_targets: the possible target values we have in this
                dataset
            possible_sensitive_attributes: the possible values of the sensitive
                attribute

        Returns:
            (dict, dict): the probabilities and the counters of each possible combination
                of target and sensitive attribute
        """
        softmax_ = F.softmax(predictions, dim=1)

        # We compute the argmax of the predictions, this is used to count
        # the number of samples for each class that are predicted with one class
        # or with the other.
        predictions_argmax = torch.argmax(torch.tensor(predictions), dim=1).to(device)

        sensitive_attribute_list = torch.tensor(
            [int(item) for item in sensitive_attribute_list]
        )
        sensitive_attribute_list = sensitive_attribute_list.to(device)

        probabilities = {}
        counters = {}
        possible_targets = [int(item) for item in possible_targets]
        possible_sensitive_attributes = [
            int(item) for item in possible_sensitive_attributes
        ]

        for z in list(possible_sensitive_attributes):
            # if we are in a binary scenario we can just consider
            # one of the two values in the computation

            target = 1
            z = int(z)
            # Z_eq_z and Z_not_eq_z are the denominators that we will use
            # in the DPL formula. |Z=z| and |Z!=z|
            Z_eq_z = len(sensitive_attribute_list[sensitive_attribute_list == z])

            Y_eq_k_and_Z_eq_z = torch.sum(
                softmax_[
                    (predictions_argmax == target) & (sensitive_attribute_list == z)
                ][:, target]
            )

            Y_eq_k_and_Z_eq_z_argmax = len(
                predictions_argmax[
                    (predictions_argmax == target) & (sensitive_attribute_list == z)
                ]
            )

            probabilities[f"{target}|{z}"] = Y_eq_k_and_Z_eq_z
            probabilities[f"{z}"] = Z_eq_z
            counters[f"{target}|{z}"] = Y_eq_k_and_Z_eq_z_argmax

            counters[f"{z}"] = Z_eq_z

        return probabilities, counters
