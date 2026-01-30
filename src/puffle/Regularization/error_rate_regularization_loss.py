import torch
import torch.nn.functional as F

from puffle.Regularization.base_fairness_loss import BaseFairnessLoss
from puffle.Regularization.formula_components import FormulaComponents
from puffle.Utils.constants import DEFAULT_ESTIMATION
from puffle.Utils.tensor_utils import ensure_tensor


class ErrorRateRegularizationLoss(BaseFairnessLoss):
    def __init__(
        self,
        weight=None,
        *,
        size_average: bool = True,
        estimation: float = DEFAULT_ESTIMATION,
    ) -> None:
        """Initialization of the regularization loss."""
        super().__init__(estimation=estimation)

    @staticmethod
    def compute_counters(
        predictions,
        true_targets,
        predictions_argmax,
        sensitive_attribute_list,
        _softmax_,
        group: int,
    ):
        """Compute error rate counters."""
        if group is None:
            msg = "The privileged and unprivileged groups must be specified"
            raise ValueError(msg)

        # Ensure inputs are tensors
        sensitive_attribute_list = ensure_tensor(
            sensitive_attribute_list, predictions.device
        )

        # Vectorized counting
        target_0 = ensure_tensor(true_targets, predictions.device) == 0
        target_1 = ensure_tensor(true_targets, predictions.device) == 1
        pred_0 = predictions_argmax == 0
        pred_1 = predictions_argmax == 1
        group_mask = sensitive_attribute_list == group

        # FP: y=0, pred=1, group=group
        fp = (target_0 & pred_1 & group_mask).sum().item()

        # TN: y=0, pred=0, group=group
        tn = (target_0 & pred_0 & group_mask).sum().item()

        # TP: y=1, pred=1, group=group
        tp = (target_1 & pred_1 & group_mask).sum().item()

        # FN: y=1, pred=0, group=group
        fn = (target_1 & pred_0 & group_mask).sum().item()

        return fp, tn, tp, fn

    @staticmethod
    def _compute_group_metric_vectorized(
        softmax_,
        predictions_argmax,
        true_targets,
        sensitive_attribute_list,
        group,
        y_val,
        pred_val,
    ):
        """Compute sum of softmax scores for a specific group, target, and prediction mask."""
        # Create mask
        # We need to handle list vs tensor for true_targets and sensitive_list
        device = softmax_.device

        t_targets = ensure_tensor(true_targets, device)
        t_sensitive = ensure_tensor(sensitive_attribute_list, device)

        # Mask: (Y=y) & (Pred=pred) & (Z=group)
        mask = (
            (t_targets == y_val)
            & (predictions_argmax == pred_val)
            & (t_sensitive == group)
        )

        if mask.sum() == 0:
            return torch.tensor(0.0, device=device), 0

        # Sum softmax scores for the predicted class
        # Note: Original logic summed softmax[:, pred_val]
        sum_val = (softmax_[:, pred_val] * mask.float()).sum()
        count = mask.sum().item()

        return sum_val, count

    @staticmethod
    def _collect_group_metrics(
        softmax_,
        predictions_argmax,
        true_targets,
        sensitive_attribute_list,
        group_id,
        group_name,
    ):
        results = {}
        cases = [(0, 1), (0, 0), (1, 1), (1, 0)]
        for y_val, pred_val in cases:
            metric_type = (
                "fp"
                if y_val == 0 and pred_val == 1
                else "tn"
                if y_val == 0 and pred_val == 0
                else "tp"
                if y_val == 1 and pred_val == 1
                else "fn"
            )
            metric_name = f"{metric_type}_{group_name}_group"

            sum_val, count = (
                ErrorRateRegularizationLoss._compute_group_metric_vectorized(
                    softmax_,
                    predictions_argmax,
                    true_targets,
                    sensitive_attribute_list,
                    group_id,
                    y_val,
                    pred_val,
                )
            )
            results[metric_name] = sum_val
            results[f"{metric_name}_argmax"] = count
        return results

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
        """
        Compute components for error rate formula.

        Note:
            Returns an empty dictionary for `analysis_dict` as it is no longer
            computed in the vectorized implementation.

        """
        if privileged_group is None or unprivileged_group is None:
            msg = "The privileged and unprivileged groups must be specified"
            raise ValueError(msg)

        results_unpriv = ErrorRateRegularizationLoss._collect_group_metrics(
            softmax_,
            predictions_argmax,
            true_targets,
            sensitive_attribute_list,
            unprivileged_group,
            "unprivileged",
        )
        results_priv = ErrorRateRegularizationLoss._collect_group_metrics(
            softmax_,
            predictions_argmax,
            true_targets,
            sensitive_attribute_list,
            privileged_group,
            "privileged",
        )

        # Merge results
        results = {**results_unpriv, **results_priv}

        # Original implementation returned analysis_dict as the 9th element.
        # We perform vectorized computation now, so we don't have this dict populated heavily.
        # We return an empty dict to maintain signature compatibility.
        analysis_dict = {}

        return FormulaComponents(
            fp_unprivileged=results["fp_unprivileged_group"],
            fp_privileged=results["fp_privileged_group"],
            tn_privileged=results["tn_privileged_group"],
            tn_unprivileged=results["tn_unprivileged_group"],
            tp_unprivileged=results["tp_unprivileged_group"],
            tp_privileged=results["tp_privileged_group"],
            fn_unprivileged=results["fn_unprivileged_group"],
            fn_privileged=results["fn_privileged_group"],
            fp_unprivileged_argmax=results["fp_unprivileged_group_argmax"],
            fp_privileged_argmax=results["fp_privileged_group_argmax"],
            tn_privileged_argmax=results["tn_privileged_group_argmax"],
            tn_unprivileged_argmax=results["tn_unprivileged_group_argmax"],
            tp_unprivileged_argmax=results["tp_unprivileged_group_argmax"],
            tp_privileged_argmax=results["tp_privileged_group_argmax"],
            fn_unprivileged_argmax=results["fn_unprivileged_group_argmax"],
            fn_privileged_argmax=results["fn_privileged_group_argmax"],
            analysis_dict=analysis_dict,
        )

    def forward(
        self,
        sensitive_attribute_list: torch.Tensor | list,
        device: torch.device,
        predictions: torch.Tensor,
        true_targets: torch.Tensor,
        possible_sensitive_attributes: list,
        possible_targets: list,
        average_probabilities: dict | None = None,
        wandb_run=None,
        batch=None,
        privileged_group=None,
        unprivileged_group=None,
        *,
        global_computation=False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Compute the regularization term.

        Args:
            sensitive_attribute_list (torch.Tensor): List of sensitive attribute values.
            device (torch.device): Computation device.
            predictions (torch.Tensor): Model predictions.
            true_targets (torch.Tensor): Ground truth labels.
            possible_sensitive_attributes (list): Possible values for sensitive attribute.
            possible_targets (list): Possible target values.
            average_probabilities (dict, optional): FL average probabilities. Defaults to None.
            wandb_run (Any, optional): WandB run for logging. Defaults to None.
            batch (Any, optional): Current batch. Defaults to None.
            privileged_group (Any, optional): Privileged group identifier. Defaults to None.
            unprivileged_group (Any, optional): Unprivileged group identifier. Defaults to None.
            global_computation (bool, optional): Whether to perform global computation. Defaults to False.

        Returns:
            torch.Tensor: The regularization loss.

        """
        if privileged_group is None or unprivileged_group is None:
            msg = "The privileged and unprivileged groups must be specified"
            raise ValueError(msg)

        (
            softmax_,
            sensitive_attribute_list,
            true_targets,
            predictions_argmax,
            possible_targets,
            possible_sensitive_attributes,
            sensitive_attribute_list_int,
        ) = self._prepare_data(
            predictions,
            sensitive_attribute_list,
            true_targets,
            device,
            possible_targets,
            possible_sensitive_attributes,
        )

        unprivileged_group_list = (
            [unprivileged_group]
            if not isinstance(unprivileged_group, (list, tuple))
            else unprivileged_group
        )
        privileged_group_list = (
            [privileged_group]
            if not isinstance(privileged_group, (list, tuple))
            else privileged_group
        )

        fairness_violations = []
        global_counters = {}

        for unprivileged in unprivileged_group_list:
            for privileged in privileged_group_list:
                violation_term, err_unpriv, err_priv = self._compute_pair_violation(
                    unprivileged,
                    privileged,
                    predictions,
                    true_targets,
                    predictions_argmax,
                    sensitive_attribute_list_int,
                    softmax_,
                    possible_sensitive_attributes,
                    average_probabilities,
                    device,
                )
                fairness_violations.append(violation_term)
                if global_computation:
                    global_counters[privileged] = err_priv
                    global_counters[unprivileged] = err_unpriv

        res = self._apply_fairness_mask(fairness_violations, device)

        if global_computation:
            return res, global_counters
        return res

    def _prepare_data(
        self,
        predictions: torch.Tensor,
        sensitive_attribute_list: torch.Tensor | list,
        true_targets: torch.Tensor,
        device: torch.device,
        possible_targets: list,
        possible_sensitive_attributes: list,
    ):
        """Prepare data for violation computation."""
        softmax_ = F.softmax(predictions, dim=1)
        sensitive_attribute_list = self._prepare_sensitive_attributes(
            sensitive_attribute_list, device
        )
        true_targets = ensure_tensor(true_targets, device)

        # We compute the argmax of the predictions
        predictions_argmax = torch.argmax(predictions.detach().clone(), dim=1).to(
            device
        )

        possible_targets = [int(item) for item in possible_targets]
        possible_sensitive_attributes = [
            int(item) for item in possible_sensitive_attributes
        ]

        sensitive_attribute_list_int = [
            item.item() if isinstance(item, torch.Tensor) else item
            for item in sensitive_attribute_list
        ]

        return (
            softmax_,
            sensitive_attribute_list,
            true_targets,
            predictions_argmax,
            possible_targets,
            possible_sensitive_attributes,
            sensitive_attribute_list_int,
        )

    def _compute_pair_violation(
        self,
        unprivileged,
        privileged,
        predictions,
        true_targets,
        predictions_argmax,
        sensitive_attribute_list_int,
        softmax_,
        possible_sensitive_attributes,
        average_probabilities,
        device,
    ):
        """Compute violation between a pair of unprivileged and privileged groups."""
        components = ErrorRateRegularizationLoss.compute_formula_components(
            predictions,
            true_targets,
            predictions_argmax,
            sensitive_attribute_list_int,
            softmax_,
            privileged_group=privileged,
            unprivileged_group=unprivileged,
        )

        fp_unpriv = components.fp_unprivileged
        fp_priv = components.fp_privileged
        tn_priv = components.tn_privileged
        tn_unpriv = components.tn_unprivileged
        tp_unpriv = components.tp_unprivileged
        tp_priv = components.tp_privileged
        fn_unpriv = components.fn_unprivileged
        fn_priv = components.fn_privileged

        err_unpriv = self._calculate_group_error_rate(
            unprivileged,
            fp_unpriv,
            tn_unpriv,
            tp_unpriv,
            fn_unpriv,
            possible_sensitive_attributes,
            average_probabilities,
        )
        err_priv = self._calculate_group_error_rate(
            privileged,
            fp_priv,
            tn_priv,
            tp_priv,
            fn_priv,
            possible_sensitive_attributes,
            average_probabilities,
        )

        if err_unpriv is None or err_priv is None:
            return torch.tensor(0.0).to(device), err_unpriv, err_priv

        error_rate = (
            err_unpriv - err_priv
            if err_unpriv - err_priv > 0
            else torch.tensor(0.0).to(device)
        )
        if not isinstance(error_rate, torch.Tensor):
            error_rate = torch.tensor(error_rate).to(device)

        return error_rate, err_unpriv, err_priv

    def _calculate_group_error_rate(
        self,
        group,
        fp,
        tn,
        tp,
        fn,
        possible_sensitive_attributes,
        average_probabilities,
    ):
        """Calculate error rate for a single group."""
        try:
            if group in possible_sensitive_attributes:
                return (fp + fn) / (fp + tn + tp + fn)
            if average_probabilities and group in average_probabilities:
                return average_probabilities[group]
        except (ZeroDivisionError, TypeError):
            pass
        return None

    def violation_with_dataset(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.DataLoader,
        average_probabilities: dict,
        device: torch.device,
        privileged_group: int,
        unprivileged_group: int,
        sum_counters: dict | None = None,
    ) -> torch.Tensor:
        """
        Compute the error rate metric on the entire dataset.

        Args:
            model (torch.nn.Module): The model to evaluate.
            dataset (torch.utils.data.DataLoader): The dataset for evaluation.
            average_probabilities (dict): Average probabilities for FL scenarios.
            device (torch.device): Computation device.
            privileged_group (int): Privileged group identifier.
            unprivileged_group (int): Unprivileged group identifier.
            sum_counters (dict, optional): Aggregated counters. Defaults to None.

        Returns:
            torch.Tensor: The error rate metric value.

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

        return self.forward(
            sensitive_attribute_list=sensitive_attribute_list,
            device=device,
            predictions=predictions,
            true_targets=torch.tensor(targets, device=device),
            possible_sensitive_attributes=sensitive_attributes,
            possible_targets=target_list,
            average_probabilities=average_probabilities,
            privileged_group=privileged_group,
            unprivileged_group=unprivileged_group,
        )

    @staticmethod
    def compute_probabilities(
        predictions,
        sensitive_attribute_list: torch.Tensor | list,
        device: torch.device,
        possible_sensitive_attributes: list,
        possible_targets: list,
        true_targets: list,
        _privileged_group: int,
        _unprivileged_group: int,
    ) -> tuple[dict, dict]:
        """
        Compute probabilities and counters.

        Args:
            predictions: Model predictions.
            sensitive_attribute_list: List of sensitive attributes.
            device (torch.device): Computation device.
            possible_sensitive_attributes (list): Possible sensitive attribute values.
            possible_targets (list): Possible target values.
            true_targets (list): Ground truth targets.
            privileged_group (int): Privileged group identifier.
            unprivileged_group (int): Unprivileged group identifier.

        """
        softmax_ = F.softmax(predictions, dim=1)

        # We compute the argmax of the predictions, this is used to count
        # the number of samples for each class that are predicted with one class
        # or with the other.
        predictions_argmax = torch.argmax(predictions.detach().clone(), dim=1).to(
            device
        )

        sensitive_attribute_list = ensure_tensor(sensitive_attribute_list, device)

        counters = {}
        possible_targets = [int(item) for item in possible_targets]
        possible_sensitive_attributes = [
            int(item) for item in possible_sensitive_attributes
        ]

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
                # z_eq_z and z_not_eq_z are the denominators that we will use
                # in the DPL formula. |Z=z| and |Z!=z|
                z_eq_z = len(sensitive_attribute_list[sensitive_attribute_list == z])

                torch.sum(
                    softmax_[
                        (predictions_argmax == target) & (sensitive_attribute_list == z)
                    ][:, target]
                )

                y_eq_k_and_z_eq_z_argmax = len(
                    predictions_argmax[
                        (predictions_argmax == target) & (sensitive_attribute_list == z)
                    ]
                )

                counters[f"{target}|{z}"] = y_eq_k_and_z_eq_z_argmax

                counters[f"{z}"] = z_eq_z

        return probabilities, counters
