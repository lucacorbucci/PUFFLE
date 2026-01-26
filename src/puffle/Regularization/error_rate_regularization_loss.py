
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ErrorRateRegularizationLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, estimation=0.5) -> None:
        """Initialization of the regularization loss."""
        super().__init__()
        self.estimation = estimation

    @staticmethod
    def compute_counters(
        predictions,
        true_targets,
        predictions_argmax,
        sensitive_attribute_list,
        softmax_,
        group: int,
    ):
        if group is None:
            msg = "The privileged and unprivileged groups must be specified"
            raise ValueError(msg)
        analysis_dict = {}
        for index, y, prediction, current_group in zip(
            list(range(len(predictions))),
            true_targets,
            predictions_argmax,
            sensitive_attribute_list, strict=False,
        ):
            prediction = int(prediction.item())
            y = int(y.item())
            if (y, prediction, current_group) not in analysis_dict:
                analysis_dict[(y, prediction, current_group)] = []
            analysis_dict[(y, prediction, current_group)].append(index)

        fp = len(analysis_dict[0, 1, group]) if (0, 1, group) in analysis_dict else 0

        tn = len(analysis_dict[0, 0, group]) if (0, 0, group) in analysis_dict else 0

        tp = len(analysis_dict[1, 1, group]) if (1, 1, group) in analysis_dict else 0

        fn = len(analysis_dict[1, 0, group]) if (1, 0, group) in analysis_dict else 0

        return (
            fp,
            tn,
            tp,
            fn,
        )

    @staticmethod
    def compute_formula_components(
        predictions,
        true_targets,
        predictions_argmax,
        sensitive_attribute_list,
        softmax_,
        privileged_group: int,
        unprivileged_group: int,
    ):
        if privileged_group is None or unprivileged_group is None:
            msg = "The privileged and unprivileged groups must be specified"
            raise ValueError(msg)

        analysis_dict = {}
        for index, y, prediction, group in zip(
            list(range(len(predictions))),
            true_targets,
            predictions_argmax,
            sensitive_attribute_list, strict=False,
        ):
            prediction = int(prediction.item())
            y = int(y.item())
            if (y, prediction, group) not in analysis_dict:
                analysis_dict[(y, prediction, group)] = []
            analysis_dict[(y, prediction, group)].append(index)

        # [gender=1 and color=1] is the privileged group
        #  [gender=0 and color=1] is the unprivileged group

        if (0, 1, unprivileged_group) in analysis_dict:
            fp_unprivileged_group = torch.sum(softmax_[analysis_dict[(0, 1, unprivileged_group)]][:, 1])
            fp_unprivileged_group_argmax = len(analysis_dict[(0, 1, unprivileged_group)])
        else:
            fp_unprivileged_group = 0
            fp_unprivileged_group_argmax = 0

        if (0, 1, privileged_group) in analysis_dict:
            fp_privileged_group = torch.sum(softmax_[analysis_dict[(0, 1, privileged_group)]][:, 1])
            fp_privileged_group_argmax = len(analysis_dict[(0, 1, privileged_group)])
        else:
            fp_privileged_group = 0
            fp_privileged_group_argmax = 0

        if (0, 0, unprivileged_group) in analysis_dict:
            tn_unprivileged_group = torch.sum(softmax_[analysis_dict[(0, 0, unprivileged_group)]][:, 0])
            tn_unprivileged_group_argmax = len(analysis_dict[(0, 0, unprivileged_group)])
        else:
            tn_unprivileged_group = 0
            tn_unprivileged_group_argmax = 0

        if (0, 0, privileged_group) in analysis_dict:
            tn_privileged_group = torch.sum(softmax_[analysis_dict[(0, 0, privileged_group)]][:, 0])
            tn_privileged_group_argmax = len(analysis_dict[(0, 0, privileged_group)])
        else:
            tn_privileged_group = 0
            tn_privileged_group_argmax = 0

        if (1, 1, unprivileged_group) in analysis_dict:
            tp_unprivileged_group = torch.sum(softmax_[analysis_dict[(1, 1, unprivileged_group)]][:, 1])
            tp_unprivileged_group_argmax = len(analysis_dict[(1, 1, unprivileged_group)])
        else:
            tp_unprivileged_group = 0
            tp_unprivileged_group_argmax = 0

        if (1, 1, privileged_group) in analysis_dict:
            tp_privileged_group = torch.sum(softmax_[analysis_dict[(1, 1, privileged_group)]][:, 1])
            tp_privileged_group_argmax = len(analysis_dict[(1, 1, privileged_group)])
        else:
            tp_privileged_group = 0
            tp_privileged_group_argmax = 0

        if (1, 0, unprivileged_group) in analysis_dict:
            fn_unprivileged_group = torch.sum(softmax_[analysis_dict[(1, 0, unprivileged_group)]][:, 0])
            fn_unprivileged_group_argmax = len(analysis_dict[(1, 0, unprivileged_group)])
        else:
            fn_unprivileged_group = 0
            fn_unprivileged_group_argmax = 0

        if (1, 0, privileged_group) in analysis_dict:
            fn_privileged_group = torch.sum(softmax_[analysis_dict[(1, 0, privileged_group)]][:, 0])
            fn_privileged_group_argmax = len(analysis_dict[(1, 0, privileged_group)])
        else:
            fn_privileged_group = 0
            fn_privileged_group_argmax = 0

        return (
            fp_unprivileged_group,
            fp_privileged_group,
            tn_privileged_group,
            tn_unprivileged_group,
            tp_unprivileged_group,
            tp_privileged_group,
            fn_unprivileged_group,
            fn_privileged_group,
            analysis_dict,
            fp_unprivileged_group_argmax,
            fp_privileged_group_argmax,
            tn_privileged_group_argmax,
            tn_unprivileged_group_argmax,
            tp_unprivileged_group_argmax,
            tp_privileged_group_argmax,
            fn_unprivileged_group_argmax,
            fn_privileged_group_argmax,
        )

    def forward(
        self,
        sensitive_attribute_list: torch.tensor,
        device: torch.device,
        predictions: torch.tensor,
        true_targets: torch.tensor,
        possible_sensitive_attributes: list,
        possible_targets: list,
        average_probabilities: dict | None = None,
        wandb_run=None,
        batch=None,
        privileged_group=None,
        unprivileged_group=None,
        global_computation=False,
    ) -> torch.tensor:
        """
        This function computes the regularization term.
        It takes as input the sensitive attribute list, the targets,
        the device and the predictions computed with the model.
        It returns the regularization term.

        What we do here:
        - We compute the softmax of the predictions
        - Then we consider the possible combinations of targets and sensitive features
            and we compute the corresponding fairness violation term
        - We return the maximum violation term among all the possible combinations

        Args:
            sensitive_attribute_list (np.array): a list with the value of
                the sensitive attribute for each sample in the batch
            device (str): the device we're using to train the model
            predictions (np.array): the predictions of the model for the batch of data
            possible_sensitive_attributes (list): the possible values of the sensitive
                attribute
            possible_targets (list): the possible target values we have in this
                dataset
            average_probabilities (dict): in case of Federated learning, if a client
                has only a subset of the possible sensitive attributes, we can use the
                average probabilities of the other clients to estimate the probabilities
                of the missing sensitive attributes. This is None in centralised learning

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
        if privileged_group is None or unprivileged_group is None:
            msg = "The privileged and unprivileged groups must be specified"
            raise ValueError(msg)
        fairness_violations = []
        # We compute the softmax of the predictions. We do this because
        # we can't use the argmax function on the nn output,
        # because we need differentiable results
        softmax_ = F.softmax(predictions, dim=1)

        # convert the list of sensitive attributes to a tensor and move it to the device
        sensitive_attribute_list = torch.tensor([int(item) for item in sensitive_attribute_list])
        sensitive_attribute_list = sensitive_attribute_list.to(device)
        true_targets = torch.tensor([int(item) for item in true_targets]).to(device)

        # We compute the argmax of the predictions, this is used to count
        # the number of samples for each class that are predicted with one class
        # or with the other.
        predictions_argmax = torch.argmax(torch.tensor(predictions), dim=1).to(device)
        # we convert the possible targets and the possible sensitive attributes to a list
        # just to be sure that the values are integers
        possible_targets = [int(item) for item in possible_targets]
        possible_sensitive_attributes = [int(item) for item in possible_sensitive_attributes]


        sensitive_attribute_list = [
            item.item() if isinstance(item, torch.Tensor) else item for item in sensitive_attribute_list
        ]

        global_counters = {}

        for unprivileged in list(unprivileged_group):
            for privileged in list(privileged_group):
                (
                    fp_unprivileged_group,
                    fp_privileged_group,
                    tn_privileged_group,
                    tn_unprivileged_group,
                    tp_unprivileged_group,
                    tp_privileged_group,
                    fn_unprivileged_group,
                    fn_privileged_group,
                    _analysis_dict,
                    _fp_unprivileged_group_argmax,
                    _fp_privileged_group_argmax,
                    _tn_privileged_group_argmax,
                    _tn_unprivileged_group_argmax,
                    _tp_unprivileged_group_argmax,
                    _tp_privileged_group_argmax,
                    _fn_unprivileged_group_argmax,
                    _fn_privileged_group_argmax,
                ) = ErrorRateRegularizationLoss.compute_formula_components(
                    predictions,
                    true_targets,
                    predictions_argmax,
                    sensitive_attribute_list,
                    softmax_,
                    privileged_group=privileged,
                    unprivileged_group=unprivileged,
                )

                added = False
                try:
                    err_unpriv = None
                    err_priv = None
                    if unprivileged in possible_sensitive_attributes:
                        err_unpriv = (fp_unprivileged_group + fn_unprivileged_group) / (
                            fp_unprivileged_group
                            + tn_unprivileged_group
                            + tp_unprivileged_group
                            + fn_unprivileged_group
                        )
                    elif average_probabilities is not None and unprivileged in average_probabilities:
                        err_unpriv = average_probabilities[unprivileged]

                    else:
                        fairness_violations.append(torch.tensor(0.0).to(device))
                        added = True

                    if privileged in possible_sensitive_attributes:
                        err_priv = (fp_privileged_group + fn_privileged_group) / (
                            fp_privileged_group + tn_privileged_group + tp_privileged_group + fn_privileged_group
                        )
                    elif average_probabilities is not None and privileged in average_probabilities:
                        err_priv = average_probabilities[privileged]
                    elif added is False:
                        fairness_violations.append(torch.tensor(0.0).to(device))

                    if err_unpriv is not None and err_priv is not None:
                        error_rate = (
                            err_unpriv - err_priv if err_unpriv - err_priv > 0 else torch.tensor(0.0).to(device)
                        )
                        if not isinstance(error_rate, torch.Tensor):
                            error_rate = torch.tensor(error_rate).to(device)
                        fairness_violations.append(error_rate)

                except Exception:
                    fairness_violations.append(torch.tensor(0.0).to(device))
                if global_computation:
                    global_counters[privileged] = err_priv

            if global_computation:
                global_counters[unprivileged] = err_unpriv

        fairness_violations_ = [item.item() if isinstance(item, torch.Tensor) else item for item in fairness_violations]

        # We get the index of the maximum violation term. Then we create a mask with
        # all zeros and we set to 1 the element at the index we found. We use this mask
        # to sum the violation terms and we return the result. This was needed because
        # when we started to work on this project we discovered that without this
        # some of the gradients were not computed correctly. I would not remove it
        # even if I'm not sure that it is needed anymore.
        index = fairness_violations_.index(max(fairness_violations_))
        fairness_violations = torch.stack(fairness_violations)
        mask = torch.full((fairness_violations.shape[0],), 0, dtype=torch.float32).to(device)
        mask[index] = 1

        res = torch.sum(mask * fairness_violations)

        if global_computation:
            return res, global_counters

        return res

    def violation_with_dataset(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.DataLoader,
        average_probabilities: dict,
        device: torch.device,
        privileged_group: int,
        unprivileged_group: int,
        sum_counters: dict | None = None,
    ) -> torch.tensor:
        """
        When we want to compute the error rate metric on the entire dataset
        we can't directly use the forward function because we don't have the
        predictions and the sensitive attribute list for each batch.
        So in this function we just use the model to compute the predictions for
        all the samples in the dataset and we aggregate the results in a single
        final tensor that we pass to the forward function.
        This is used, for instance, to compute the error rate of the model
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
            float: the error rate metric computed on the dataset
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
                sensitive_attribute_list = torch.cat((sensitive_attribute_list, sensitive_attributes.to(device)), 0)
                targets += target.tolist()

        sensitive_attributes = list({item.item() for item in sensitive_attribute_list})
        target_list = list(set(targets))

        return self.forward(
            sensitive_attribute_list=sensitive_attribute_list,
            device=device,
            predictions=predictions,
            true_targets=np.array(targets),
            possible_sensitive_attributes=sensitive_attributes,
            possible_targets=target_list,
            average_probabilities=average_probabilities,
            privileged_group=privileged_group,
            unprivileged_group=unprivileged_group,
        )

    def compute_violation_with_argmax(
        self,
        predictions_argmax: torch.tensor,
        sensitive_attribute_list: torch.tensor,
        y_true: torch.tensor,
        analysis_dict: dict,
    ):
        """
        Debug function used to compute the DPL using the argmax function
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
        total_num_samples = len(sensitive_attribute_list)

        return max(
            abs(
                (len(analysis_dict[(1, 0, 1.0)]) / total_num_samples)
                - (len(analysis_dict[(1, 0, 0.0)]) / total_num_samples)
            ),
            abs(
                (len(analysis_dict[(1, 1, 1.0)]) / total_num_samples)
                - (len(analysis_dict[(1, 1, 0.0)]) / total_num_samples)
            ),
        )

    @staticmethod
    def compute_probabilities(
        predictions,
        sensitive_attribute_list,
        device: torch.device,
        possible_sensitive_attributes: list,
        possible_targets: list,
        true_targets: list,
        privileged_group: int,
        unprivileged_group: int,
    ) -> torch.tensor:
        """
        This function computes the probabilities and the counters
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

        sensitive_attribute_list = torch.tensor([int(item) for item in sensitive_attribute_list])
        sensitive_attribute_list = sensitive_attribute_list.to(device)

        counters = {}
        possible_targets = [int(item) for item in possible_targets]
        possible_sensitive_attributes = [int(item) for item in possible_sensitive_attributes]

        counters = {}
        probabilities = {}

        for group in possible_sensitive_attributes:
            fp, tn, tp, fn = ErrorRateRegularizationLoss.compute_counters(
                predictions,
                true_targets,
                predictions_argmax,
                [int(item) for item in sensitive_attribute_list],
                softmax_,
                group=int(group),
            )

            if f"{group}_fp" not in counters:
                counters[f"{group}_fp"] = fp
                counters[f"{group}_tn"] = tn
                counters[f"{group}_tp"] = tp
                counters[f"{group}_fn"] = fn
                probabilities[f"{group}_denominator"] = fp + fn + tp + tn
                probabilities[f"{group}_numerator"] = fp + fn
            else:
                counters[f"{group}_fp"] += fp
                counters[f"{group}_tn"] += tn
                counters[f"{group}_tp"] += tp
                counters[f"{group}_fn"] += fn
        for target in possible_targets:
            for group in possible_sensitive_attributes:
                z = int(group)
                # Z_eq_z and Z_not_eq_z are the denominators that we will use
                # in the DPL formula. |Z=z| and |Z!=z|
                Z_eq_z = len(sensitive_attribute_list[sensitive_attribute_list == z])

                torch.sum(
                    softmax_[(predictions_argmax == target) & (sensitive_attribute_list == z)][:, target]
                )

                Y_eq_k_and_Z_eq_z_argmax = len(
                    predictions_argmax[(predictions_argmax == target) & (sensitive_attribute_list == z)]
                )

                counters[f"{target}|{z}"] = Y_eq_k_and_Z_eq_z_argmax

                counters[f"{z}"] = Z_eq_z

        return probabilities, counters
