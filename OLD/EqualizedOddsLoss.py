import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# from .DPL.DPLUtilsutils import Utils


class EqualizedOddsLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, estimation=0.5) -> None:
        """Initialization of the regularization loss."""
        super().__init__()
        self.estimation = estimation

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
            raise ValueError("The privileged and unprivileged groups must be specified")

        analysis_dict = {}
        for index, y, prediction, group in zip(
            list(range(len(predictions))),
            true_targets,
            predictions_argmax,
            sensitive_attribute_list,
        ):
            prediction = int(prediction.item())
            y = int(y.item())
            if (y, prediction, group) not in analysis_dict:
                analysis_dict[(y, prediction, group)] = []
            analysis_dict[(y, prediction, group)].append(index)

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
        privileged_group: int = None,
        unprivileged_group: int = None,
        average_probabilities: dict = None,
        wandb_run=None,
        batch=None,
    ) -> torch.tensor:
        """This function computes the regularization term.
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
            raise ValueError("The privileged and unprivileged groups must be specified")
        fairness_violations = []
        fairness_violations_with_argmax = []
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

        analysis_dict = {}

        sensitive_attribute_list = [1 if item == 1.0 else 0 for item in sensitive_attribute_list]

        (
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
        ) = EqualizedOddsLoss.compute_formula_components(
            predictions,
            true_targets,
            predictions_argmax,
            sensitive_attribute_list,
            softmax_,
            privileged_group=privileged_group,
            unprivileged_group=unprivileged_group,
        )

        if wandb_run:
            wandb_run.log(
                {
                    "batch": batch,
                    "Counter FP Male": fp_unprivileged_group,
                    "Counter FP Female": fp_privileged_group,
                    "Counter TN Male": tn_unprivileged_group,
                    "Counter TN Female": tn_privileged_group,
                    "Counter TP Male": tp_unprivileged_group,
                    "Counter TP Female": tp_privileged_group,
                    "Counter FN Male": fn_unprivileged_group,
                    "Counter FN Female": fn_privileged_group,
                    "Counter FP Male Argmax": fp_unprivileged_group_argmax,
                    "Counter FP Female Argmax": fp_privileged_group_argmax,
                    "Counter TN Male Argmax": tn_unprivileged_group_argmax,
                    "Counter TN Female Argmax": tn_privileged_group_argmax,
                    "Counter TP Male Argmax": tp_unprivileged_group_argmax,
                    "Counter TP Female Argmax": tp_privileged_group_argmax,
                    "Counter FN Male Argmax": fn_unprivileged_group_argmax,
                    "Counter FN Female Argmax": fn_privileged_group_argmax,
                }
            )

        try:
            # fairness_violations.append(
            #     max(
            #         # FPR
            #         abs(
            #             (fp_male / (fp_male + tn_male))
            #             - (fp_female / (fp_female + tn_female))
            #         ),
            #         # TPR
            #         abs(
            #             (tp_male / (tp_male + fn_male))
            #             - (tp_female / (tp_female + fn_female))
            #         ),
            #     )
            # )

            # fpr = (tp_male / (tp_male + fn_male)) - (tp_female / (tp_female + fn_female))

            # tpr = (fp_male / (fp_male + tn_male)) - (fp_female / (fp_female + tn_female))

            # if fpr < 0:
            #     fpr = ((tp_male / (tp_male + fn_male)) - (tp_male / (tp_male + fn_male))) + ((tp_female / (tp_female + fn_female))-(tp_female / (tp_female + fn_female)))
            # if tpr < 0:
            #     tpr = ((fp_male / (fp_male + tn_male)) - (fp_male / (fp_male + tn_male))) + ((fp_female / (fp_female + tn_female))-(fp_female / (fp_female + tn_female)))

            # fairness_violations.append(fpr)
            # fairness_violations.append(tpr)

            # tpr[unpriv] - tpr[priv]
            tpr = (tp_unprivileged_group / (tp_unprivileged_group + fn_unprivileged_group)) - (
                tp_privileged_group / (tp_privileged_group + fn_privileged_group)
            )
            # fpr[unpriv] - fpr[priv]
            fpr = (fp_unprivileged_group / (fp_unprivileged_group + tn_unprivileged_group)) - (
                fp_privileged_group / (fp_privileged_group + tn_privileged_group)
            )

            if fpr > 0 and tpr > 0:
                # I need regularization
                fairness_violations.append(fpr)
            elif fpr < 0 and tpr < 0:
                # I need regularization
                fairness_violations.append(abs(tpr))
            elif fpr > 0 and tpr < 0:
                # I need regularization
                fairness_violations.append(abs(fpr))
                fairness_violations.append(abs(tpr))
            else:
                # I don't need regularization
                fairness_violations.append(torch.tensor(0.0).to(device))

            # if tpr > 0:
            #     # tpr is 0
            #     tpr = ((fp_unprivileged_group / (fp_unprivileged_group + tn_unprivileged_group)) - (fp_unprivileged_group / (fp_unprivileged_group + tn_unprivileged_group))) + ((fp_privileged_group / (fp_privileged_group + tn_privileged_group))-(fp_privileged_group / (fp_privileged_group + tn_privileged_group)))

            # if fpr > 0:
            #     # fpr is 0
            #     fpr = ((tp_unprivileged_group / (tp_unprivileged_group + fn_unprivileged_group)) - (tp_unprivileged_group / (tp_unprivileged_group + fn_unprivileged_group))) + ((tp_privileged_group / (tp_privileged_group + fn_privileged_group))-(tp_privileged_group / (tp_privileged_group + fn_privileged_group)))

            # fairness_violations.append(fpr)
            # fairness_violations.append(tpr)

            # fairness_violations.append(
            #         abs(
            #             (tp_unprivileged_group / (tp_unprivileged_group + fn_unprivileged_group))
            #             - (tp_privileged_group / (tp_privileged_group + fn_privileged_group))
            #         ),
            # )
            # fairness_violations.append(
            #         abs(
            #             (fp_unprivileged_group / (fp_unprivileged_group + tn_unprivileged_group))
            #             - (fp_privileged_group / (fp_privileged_group + tn_privileged_group))
            #         )
            # )

            # fairness_violations_with_argmax = max(
            #     # TPR
            #     abs(
            #             (tp_unprivileged_group_argmax / (tp_unprivileged_group_argmax + fn_unprivileged_group_argmax))
            #             - (tp_privileged_group_argmax / (tp_privileged_group_argmax + fn_privileged_group_argmax))
            #         ),
            #     # FPR
            #     abs(
            #             (fp_unprivileged_group_argmax / (fp_unprivileged_group_argmax + tn_unprivileged_group_argmax))
            #             - (fp_privileged_group_argmax / (fp_privileged_group_argmax + tn_privileged_group_argmax))
            #         ),
            # )
            # if wandb_run:
            #     wandb_run.log(
            #         {
            #             "batch": batch,
            #             "Batch Equalized Odds with argmax": int(
            #                 fairness_violations_with_argmax
            #             ),
            #         }
            #     )
        except:
            return None

        # return fairness_violations[0]

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

        return res

    def violation_with_dataset(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.DataLoader,
        average_probabilities: dict,
        device: torch.device,
        privileged_group: int,
        unprivileged_group: int,
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
            for images, sensitive_attributes, target in dataset:
                images = images.to(device)
                target = target.to(device)

                output = model(images)

                predictions = torch.cat((predictions, output), 0)
                sensitive_attribute_list = torch.cat((sensitive_attribute_list, sensitive_attributes.to(device)), 0)
                targets += target.tolist()

        sensitive_attributes = list({item.item() for item in sensitive_attribute_list})
        target_list = list(set(targets))

        # now we just call the forward function with the "fake" predictions and the sensitive
        # attribute list we computed
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
        # current_target: int,
        # current_sensitive_feature: int,
        # weights: dict = None,
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

        sensitive_attribute_list = torch.tensor([int(item) for item in sensitive_attribute_list])
        sensitive_attribute_list = sensitive_attribute_list.to(device)

        probabilities = {}
        counters = {}
        possible_targets = [int(item) for item in possible_targets]
        possible_sensitive_attributes = [int(item) for item in possible_sensitive_attributes]

        (
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
        ) = EqualizedOddsLoss.compute_formula_components(
            predictions,
            true_targets,
            predictions_argmax,
            [int(item) for item in sensitive_attribute_list],
            softmax_,
            privileged_group=privileged_group,
            unprivileged_group=unprivileged_group,
        )
        counters = {
            "fp_unprivileged_argmax": fp_unprivileged_group_argmax,
            "fp_privileged_argmax": fp_privileged_group_argmax,
            "tn_privileged_argmax": tn_privileged_group_argmax,
            "tn_unprivileged_argmax": tn_unprivileged_group_argmax,
            "tp_unprivileged_argmax": tp_unprivileged_group_argmax,
            "tp_privileged_argmax": tp_privileged_group_argmax,
            "fn_unprivileged_argmax": fn_unprivileged_group_argmax,
            "fn_privileged_argmax": fn_privileged_group_argmax,
            "fp_unprivileged": fp_unprivileged_group,
            "fp_privileged": fp_privileged_group,
            "tn_privileged": tn_privileged_group,
            "tn_unprivileged": tn_unprivileged_group,
            "tp_unprivileged": tp_unprivileged_group,
            "tp_privileged": tp_privileged_group,
            "fn_unprivileged": fn_unprivileged_group,
            "fn_privileged": fn_privileged_group,
        }

        for z in list(possible_sensitive_attributes):
            # if we are in a binary scenario we can just consider
            # one of the two values in the computation

            target = 1
            z = int(z)
            # Z_eq_z and Z_not_eq_z are the denominators that we will use
            # in the DPL formula. |Z=z| and |Z!=z|
            Z_eq_z = len(sensitive_attribute_list[sensitive_attribute_list == z])

            Y_eq_k_and_Z_eq_z = torch.sum(
                softmax_[(predictions_argmax == target) & (sensitive_attribute_list == z)][:, target]
            )

            Y_eq_k_and_Z_eq_z_argmax = len(
                predictions_argmax[(predictions_argmax == target) & (sensitive_attribute_list == z)]
            )

            probabilities[f"{target}|{z}"] = Y_eq_k_and_Z_eq_z
            probabilities[f"{z}"] = Z_eq_z
            # counters[f"{target}|{z}"] = Y_eq_k_and_Z_eq_z_argmax

            # counters[f"{z}"] = Z_eq_z

        return probabilities, counters
