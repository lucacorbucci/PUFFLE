import torch
import torch.nn.functional as F

from puffle.Regularization.base_fairness_loss import BaseFairnessLoss
from puffle.Utils.constants import DEFAULT_ESTIMATION
from puffle.Utils.tensor_utils import ensure_tensor


class DisparityRegularizationLoss(BaseFairnessLoss):
    """
    Defines the regularization loss as proposed in
    https://arxiv.org/abs/2302.09183.
    It uses the definition of demographic parity to compute the
    fairness violation term for each batch and then it uses this
    violation term as a regularization term to add to the loss.
    """

    def __init__(
        self,
        _weight=None,
        *,
        _size_average: bool = True,
        estimation: float = DEFAULT_ESTIMATION,
    ) -> None:
        """Initialization of the regularization loss."""
        super().__init__(estimation=estimation)

    def forward(
        self,
        sensitive_attribute_list: torch.Tensor | list,
        device: torch.device,
        predictions: torch.Tensor,
        possible_sensitive_attributes: list,
        possible_targets: list,
        average_probabilities: dict | None = None,
        _wandb_run=None,
        _batch=None,
        *,
        global_computation=False,
        json_file=None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Compute the regularization term.

        Args:
            sensitive_attribute_list (torch.Tensor): List of sensitive group indicators.
            device (torch.device): Computation device.
            predictions (torch.Tensor): Model output predictions.
            possible_sensitive_attributes (list): List of possible sensitive group values.
            possible_targets (list): List of possible target labels.
            average_probabilities (dict, optional): FL average probabilities. Defaults to None.
            wandb_run (Any, optional): WandB run for logging. Defaults to None.
            batch (Any, optional): Current batch. Defaults to None.
            global_computation (bool, optional): Whether to perform global computation. Defaults to False.
            json_file (str, optional): Path to JSON file for logging. Defaults to None.

        Returns:
            torch.Tensor: The disparity regularization loss.

        """
        (
            softmax_,
            sensitive_attribute_list,
            predictions_argmax,
            possible_targets,
            possible_sensitive_attributes,
        ) = self._prepare_data(
            predictions,
            sensitive_attribute_list,
            device,
            possible_targets,
            possible_sensitive_attributes,
        )

        fairness_violations = []
        global_counters = {}

        for target in possible_targets:
            for z in possible_sensitive_attributes:
                violation_term, y_eq_k_and_z_eq_z = self._compute_violation_term(
                    target,
                    z,
                    softmax_,
                    predictions_argmax,
                    sensitive_attribute_list,
                    average_probabilities,
                )
                fairness_violations.append(violation_term)
                global_counters[f"{target}|{z}"] = y_eq_k_and_z_eq_z

        res = self._apply_fairness_mask(fairness_violations, device)

        if global_computation:
            global_counters = self._update_global_counters(global_counters, json_file)
            return (res, global_counters)
        return res

    def _prepare_data(
        self,
        predictions: torch.Tensor,
        sensitive_attribute_list: torch.Tensor | list,
        device: torch.device,
        possible_targets: list,
        possible_sensitive_attributes: list,
    ):
        """Prepare data for disparity computation."""
        softmax_ = F.softmax(predictions, dim=1)
        sensitive_attribute_list = self._prepare_sensitive_attributes(
            sensitive_attribute_list, device
        )

        # We compute the argmax of the predictions
        predictions_argmax = torch.argmax(predictions.detach().clone(), dim=1).to(
            device
        )

        possible_targets = [int(item) for item in possible_targets]
        possible_sensitive_attributes = [
            int(item) for item in possible_sensitive_attributes
        ]

        return (
            softmax_,
            sensitive_attribute_list,
            predictions_argmax,
            possible_targets,
            possible_sensitive_attributes,
        )

    def _compute_violation_term(
        self,
        target,
        z,
        softmax_,
        predictions_argmax,
        sensitive_attribute_list,
        average_probabilities,
    ):
        """Compute violation term for a specific target and sensitive group."""
        # Dennominators |z=z| and |z!=z|
        z_eq_z = len(sensitive_attribute_list[sensitive_attribute_list == z])
        z_not_eq_z = len(sensitive_attribute_list[sensitive_attribute_list != z])

        # |Y = k, Z = z|
        y_eq_k_and_z_eq_z = torch.sum(
            softmax_[(predictions_argmax == target) & (sensitive_attribute_list == z)][
                :, target
            ]
        )

        # |Y = k, Z != z|
        y_eq_k_and_z_not_eq_z = torch.sum(
            softmax_[(predictions_argmax == target) & (sensitive_attribute_list != z)][
                :, target
            ]
        )

        if (y_eq_k_and_z_eq_z == 0 and y_eq_k_and_z_not_eq_z != 0) or (
            z_eq_z == 0 and z_not_eq_z != 0
        ):
            violation_term = self._estimate_violation(
                target,
                z,
                y_eq_k_and_z_not_eq_z,
                z_not_eq_z,
                average_probabilities,
                is_z_zero=True,
            )
        elif (y_eq_k_and_z_eq_z != 0 and y_eq_k_and_z_not_eq_z == 0) or (
            z_not_eq_z == 0 and z_eq_z != 0
        ):
            violation_term = self._estimate_violation(
                target,
                z,
                y_eq_k_and_z_eq_z,
                z_eq_z,
                average_probabilities,
                is_z_zero=False,
            )
        else:
            violation_term = torch.abs(
                (y_eq_k_and_z_eq_z / z_eq_z) - (y_eq_k_and_z_not_eq_z / z_not_eq_z)
            )

        return violation_term, y_eq_k_and_z_eq_z

    def _estimate_violation(
        self,
        target,
        z,
        known_numerator,
        known_denominator,
        average_probabilities,
        *,
        is_z_zero: bool,
    ):
        """Estimate violation term when one group is missing."""
        denominator_val = (1 if z == 1 else 0) if is_z_zero else (1 if z == 0 else 0)
        prob_key = f"{target}|{denominator_val}"

        if average_probabilities and average_probabilities.get(prob_key) is not None:
            if is_z_zero:
                return abs(
                    average_probabilities[prob_key]
                    - known_numerator / known_denominator
                )
            return abs(
                (known_numerator / known_denominator) - average_probabilities[prob_key]
            )

        # Default fallback: return 0 if no estimation available
        val = known_numerator / known_denominator
        return torch.abs(val) - torch.abs(val)

    def _update_global_counters(self, global_counters, json_file):
        """Update global counters based on json_file info."""
        if not json_file:
            return global_counters

        for sens_value in json_file.get("possible_z", []):
            poss_target = 0
            try:
                global_counters[sens_value] = global_counters.get(
                    f"{poss_target}|{sens_value}", 0
                ) + global_counters.get(f"{abs(1 - poss_target)}|{sens_value}", 0)
            except (KeyError, ValueError, TypeError):
                continue

        for non_existing, _ in json_file.get("missing_combinations", []):
            global_counters.pop(non_existing, None)

        return global_counters

    def violation_with_dataset(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.DataLoader,
        average_probabilities: dict,
        device: torch.device,
        *,
        global_computation=False,
    ) -> torch.Tensor:
        """
        Evaluate the violation term on the entire dataset.

        Args:
            model (torch.nn.Module): The model to evaluate.
            dataset (torch.utils.data.DataLoader): The dataset for evaluation.
            average_probabilities (dict): FL average probabilities.
            device (torch.device): Computation device.
            global_computation (bool, optional): Whether to perform global computation. Defaults to False.

        """
        predictions = torch.tensor([]).to(device)
        sensitive_attribute_list = torch.tensor([]).to(device)
        targets = []
        model.eval()
        with torch.no_grad():
            for images_batch, sensitive_attributes_batch, target_batch in dataset:
                images_batch = images_batch.to(device)
                target_batch = target_batch.to(device)

                output = model(images_batch)

                predictions = torch.cat((predictions, output), 0)
                sensitive_attribute_list = torch.cat(
                    (sensitive_attribute_list, sensitive_attributes_batch.to(device)), 0
                )
                targets += target_batch.tolist()

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
            global_computation=global_computation,
        )

    def evaluate_violation(
        self,
        predictions_argmax: torch.Tensor,
        sensitive_attribute_list: torch.Tensor | list,
        possible_sensitive_attributes: list,
        possible_targets: list,
        *,
        global_computation=False,
    ) -> torch.Tensor:
        """
        Compute violation using argmax.

        Args:
            predictions_argmax (torch.Tensor): Predictions (argmax).
            sensitive_attribute_list (torch.Tensor): Sensitive attributes.
            possible_sensitive_attributes (list): List of possible sensitive feature values.
            possible_targets (list): List of possible target labels.
            global_computation (bool, optional): Whether to perform global computation. Defaults to False.

        Returns:
            torch.Tensor: The disparity regularization loss.

        """
        fairness_violations = []
        for target in possible_targets:
            for z in possible_sensitive_attributes:
                violation_term = self.compute_violation_with_argmax(
                    predictions_argmax, sensitive_attribute_list, target, z
                )
                fairness_violations.append(violation_term)

        fairness_violations_ = []
        for item in fairness_violations:
            if isinstance(item, torch.Tensor):
                if item.numel() > 1:
                    fairness_violations_.append(item.mean().item())
                else:
                    fairness_violations_.append(item.item())
            else:
                fairness_violations_.append(item)

        index = fairness_violations_.index(max(fairness_violations_))
        fairness_violations_tensors = []
        for item in fairness_violations:
            if isinstance(item, torch.Tensor):
                if item.numel() > 1:
                    fairness_violations_tensors.append(item.mean())
                else:
                    fairness_violations_tensors.append(item)
            else:
                fairness_violations_tensors.append(
                    torch.tensor(item, dtype=torch.float32).to(
                        predictions_argmax.device
                    )
                )

        fairness_violations = torch.stack(fairness_violations_tensors)

        mask = torch.full((fairness_violations.shape[0],), 0, dtype=torch.float32).to(
            predictions_argmax.device
        )
        mask[index] = 1
        mask = mask.to(predictions_argmax.device)
        fairness_violations = fairness_violations.to(predictions_argmax.device)
        return torch.sum(mask * fairness_violations)

    def compute_violation_with_argmax(
        self,
        predictions_argmax: torch.Tensor,
        sensitive_attribute_list: torch.Tensor | list,
        current_target: int,
        current_sensitive_feature: int,
        weights: dict | None = None,
    ):
        """
        Compute violation using argmax.

        Args:
            predictions_argmax (torch.Tensor): Predictions (argmax).
            sensitive_attribute_list (torch.Tensor): Sensitive attributes.
            current_target (int): Target being considered.
            current_sensitive_feature (int): Sensitive feature value being considered.
            weights (dict, optional): Weights for each sample. Defaults to None.

        Returns:
            Tuple[int, int, int, int]: Counts for violation calculation.

        """
        # Z_eq_z and Z_not_eq_z are the denominators that we will use
        # in the DPL formula. |Z=z| and |Z!=z|

        z_eq_z = len(
            sensitive_attribute_list[
                sensitive_attribute_list == current_sensitive_feature
            ]
        )
        z_not_eq_z = len(
            sensitive_attribute_list[
                sensitive_attribute_list != current_sensitive_feature
            ]
        )
        y_eq_k_and_z_eq_z = len(
            predictions_argmax[
                (predictions_argmax == current_target)
                & (sensitive_attribute_list == current_sensitive_feature)
            ]
        )

        y_eq_k_and_z_not_eq_z = len(
            predictions_argmax[
                (predictions_argmax == current_target)
                & (sensitive_attribute_list != current_sensitive_feature)
            ]
        )

        if z_eq_z == 0 and z_not_eq_z != 0:
            return abs(y_eq_k_and_z_not_eq_z / z_not_eq_z)
        if z_eq_z != 0 and z_not_eq_z == 0:
            return abs(y_eq_k_and_z_eq_z / z_eq_z)
        if z_eq_z == 0 and z_not_eq_z == 0:
            return 0.0
        return abs(y_eq_k_and_z_eq_z / z_eq_z - y_eq_k_and_z_not_eq_z / z_not_eq_z)

    @staticmethod
    def compute_probabilities(
        predictions: torch.Tensor,
        sensitive_attribute_list: torch.Tensor | list,
        device: torch.device,
        possible_sensitive_attributes: list,
        possible_targets: list,
    ) -> tuple[dict, dict]:
        """
        Compute the probabilities and the counters
            of each possible combination of target and sensitive attribute.

        Args:
            predictions (torch.Tensor): Model predictions.
            sensitive_attribute_list (torch.Tensor): Sensitive attribute values.
            device (torch.device): Computation device.
            possible_sensitive_attributes (list): List of possible sensitive attributes.
            possible_targets (list): List of possible targets.

        Returns:
            (dict, dict): The probabilities and the counters of each possible combination.

        """
        softmax_ = F.softmax(predictions, dim=1)

        # We compute the argmax of the predictions, this is used to count
        # the number of samples for each class that are predicted with one class
        # or with the other.
        predictions_argmax = torch.argmax(predictions.detach().clone(), dim=1).to(
            device
        )

        sensitive_attribute_list = ensure_tensor(sensitive_attribute_list, device)

        probabilities = {}
        counters = {}
        possible_sensitive_attributes = [
            int(item) for item in possible_sensitive_attributes
        ]

        for target in possible_targets:
            for z in list(possible_sensitive_attributes):
                # if we are in a binary scenario we can just consider
                # one of the two values in the computation

                z_int = int(z)
                # z_eq_z and z_not_eq_z are the denominators that we will use
                # in the DPL formula. |Z=z| and |Z!=z|
                z_eq_z = len(
                    sensitive_attribute_list[sensitive_attribute_list == z_int]
                )

                y_eq_k_and_z_eq_z = torch.sum(
                    softmax_[
                        (predictions_argmax == target) & (sensitive_attribute_list == z)
                    ][:, target]
                )

                y_eq_k_and_z_eq_z_argmax = len(
                    predictions_argmax[
                        (predictions_argmax == target) & (sensitive_attribute_list == z)
                    ]
                )

                probabilities[f"{target}|{z}"] = y_eq_k_and_z_eq_z
                probabilities[f"{z}"] = z_eq_z
                counters[f"{target}|{z}"] = y_eq_k_and_z_eq_z_argmax

                counters[f"{z}"] = z_eq_z

        return probabilities, counters
