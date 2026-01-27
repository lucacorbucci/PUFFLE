from typing import Any, cast

import pytest
import torch

from puffle.Regularization.error_rate_regularization_loss import (
    ErrorRateRegularizationLoss,
)


class TestErrorRateCoverage:
    def test_compute_counters_errors(self):
        # Missing group
        with pytest.raises(ValueError, match="groups must be specified"):
            ErrorRateRegularizationLoss.compute_counters(
                cast("Any", None),
                cast("Any", None),
                cast("Any", None),
                cast("Any", None),
                cast("Any", None),
                group=cast("Any", None),
            )

    def test_compute_formula_components_errors(self):
        # Missing groups
        with pytest.raises(ValueError, match="groups must be specified"):
            ErrorRateRegularizationLoss.compute_formula_components(
                cast("Any", None),
                cast("Any", None),
                cast("Any", None),
                cast("Any", None),
                cast("Any", None),
                privileged_group=cast("Any", None),
                unprivileged_group=1,
            )

        with pytest.raises(ValueError, match="groups must be specified"):
            ErrorRateRegularizationLoss.compute_formula_components(
                cast("Any", None),
                cast("Any", None),
                cast("Any", None),
                cast("Any", None),
                cast("Any", None),
                privileged_group=0,
                unprivileged_group=cast("Any", None),
            )

    def test_forward_errors(self):
        loss = ErrorRateRegularizationLoss()
        with pytest.raises(ValueError, match="groups must be specified"):
            loss.forward(
                sensitive_attribute_list=[],
                device=torch.device("cpu"),
                predictions=torch.tensor([]),
                true_targets=torch.tensor([]),
                possible_sensitive_attributes=[],
                possible_targets=[],
                privileged_group=cast("Any", None),
                unprivileged_group=1,
            )

    def test_calculate_group_error_rate_zero_division(self):
        loss = ErrorRateRegularizationLoss()
        # fp=0, tn=0, tp=0, fn=0 -> ZeroDivisionError logic check
        # The method catches ZeroDivisionError/TypeError and returns None
        rate = loss._calculate_group_error_rate(
            group=0,
            fp=0,
            tn=0,
            tp=0,
            fn=0,
            possible_sensitive_attributes=[0],
            average_probabilities=None,
        )
        assert rate is None

    def test_compute_probabilities(self):
        # Cover compute_probabilities basic run using duplicates to trigger accumulation
        preds = torch.tensor([[0.8, 0.2], [0.8, 0.2], [0.3, 0.7]])
        sens = [0, 0, 1]
        true_targets = [0, 0, 1]
        device = torch.device("cpu")

        probs, counters = ErrorRateRegularizationLoss.compute_probabilities(
            preds,
            sens,
            device,
            possible_sensitive_attributes=[0, 0, 1],
            possible_targets=[0, 1],
            true_targets=true_targets,
            _privileged_group=0,
            _unprivileged_group=1,
        )

        assert isinstance(probs, dict)
        assert isinstance(counters, dict)

    def test_compute_counters_list_input(self):
        # Trigger list conversion path
        preds = torch.tensor([[0.8, 0.2]])

        # compute_counters implementation:
        # if isinstance(sensitive_attribute_list, list): convert...
        fp, _tn, _tp, _fn = ErrorRateRegularizationLoss.compute_counters(
            predictions=preds,
            true_targets=[0],
            predictions_argmax=torch.tensor([0]),
            sensitive_attribute_list=[0],
            _softmax_=preds,
            group=0,
        )
        assert fp == 0

    def test_forward_global_computation(self):
        loss = ErrorRateRegularizationLoss()
        # Trigger global_computation branches
        preds = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        device = torch.device("cpu")
        res, counters = loss.forward(
            sensitive_attribute_list=[0, 1],
            device=device,
            predictions=preds,
            true_targets=torch.tensor([0, 1]),
            possible_sensitive_attributes=[0, 1],
            possible_targets=[0, 1],
            privileged_group=0,
            unprivileged_group=1,
            global_computation=True,
        )
        assert isinstance(res, torch.Tensor)
        assert isinstance(counters, dict)
        assert 0 in counters
        assert 1 in counters

    def test_calculate_group_error_rate_none(self):
        loss = ErrorRateRegularizationLoss()
        # group not in possible_sensitive_attributes and no avg_probs
        rate = loss._calculate_group_error_rate(
            group=99,
            fp=0,
            tn=0,
            tp=0,
            fn=0,
            possible_sensitive_attributes=[0, 1],
            average_probabilities=None,
        )
        assert rate is None

        # Test the branch where err_unpriv is None in _compute_pair_violation
        with torch.no_grad():
            violation, err_u, _err_p = loss._compute_pair_violation(
                unprivileged=cast("Any", 99),
                privileged=cast("Any", 0),
                predictions=torch.tensor([[0.8, 0.2]]),
                true_targets=torch.tensor([0]),
                predictions_argmax=torch.tensor([0]),
                sensitive_attribute_list_int=[99],
                softmax_=torch.tensor([[0.8, 0.2]]),
                possible_sensitive_attributes=[0],  # 99 not here
                average_probabilities=None,
                device=torch.device("cpu"),
            )
            assert violation.item() == 0.0
            assert err_u is None

    def test_apply_fairness_mask_list_of_tensors(self):
        loss = ErrorRateRegularizationLoss()
        # Trigger branches in _apply_fairness_mask where violations are mixtures
        violations = [torch.tensor([0.5, 0.6]), 0.7]  # tensor with numel > 1 and float
        res = loss._apply_fairness_mask(violations, torch.device("cpu"))
        # max of [0.55, 0.7] is 0.7.
        assert res.item() == pytest.approx(0.7)
