import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from puffle.Regularization.error_rate_regularization_loss import (
    ErrorRateRegularizationLoss,
)


class TestErrorRateLoss:
    def test_compute_probabilities_output_type(self):
        # Test if the output is a tuple containing two dictionaries
        predictions = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        sensitive_attribute_list = [0, 1]
        device = torch.device("cpu")
        possible_sensitive_attributes = [0, 1]
        possible_targets = [0, 1]
        true_targets = [0, 1]
        privileged_group = 0
        unprivileged_group = 1

        result = ErrorRateRegularizationLoss.compute_probabilities(
            predictions,
            sensitive_attribute_list,
            device,
            possible_sensitive_attributes,
            possible_targets,
            true_targets,
            privileged_group,
            unprivileged_group,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(item, dict) for item in result)

    def test_compute_counters(self):
        # Test compute_counters static method
        predictions = torch.tensor([[0.8, 0.2], [0.1, 0.9], [0.7, 0.3]])
        true_targets = [0, 1, 0]
        predictions_argmax = torch.tensor([0, 1, 0])
        sensitive_attribute_list = [1, 1, 0]
        softmax_ = torch.softmax(predictions, dim=1)
        group = 1

        # for group 1: samples index 0 and 1
        # sample 0: true=0, pred=0 -> TP (for class 0) or TN (depending on target)
        # However, compute_counters computes fp, tn, tp, fn for a specific group relative to error rates.
        # Let's check implementation.

        fp, tn, tp, fn = ErrorRateRegularizationLoss.compute_counters(
            predictions,
            true_targets,
            predictions_argmax,
            sensitive_attribute_list,
            softmax_,
            group=group,
        )

        # Sample 0: true=0, pred=0, group=1 -> TN
        # Sample 1: true=1, pred=1, group=1 -> TP
        # Sample 2: true=0, pred=0, group=0 -> (Not group 1)

        # Expected: fp=0, tn=1, tp=1, fn=0

        fp, tn, tp, fn = ErrorRateRegularizationLoss.compute_counters(
            predictions,
            true_targets,
            predictions_argmax,
            [int(x) for x in sensitive_attribute_list],  # Ensure integers
            softmax_,
            group=group,
        )

        assert fp == 0
        assert tn == 1
        assert tp == 1
        assert fn == 0

    def test_forward_basic(self):
        # Test the forward method
        predictions = torch.tensor(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.4, 0.6],
            ]
        )
        sensitive_attribute_list = [0, 1, 0, 1]
        device = torch.device("cpu")
        possible_sensitive_attributes = [0, 1]
        possible_targets = [0, 1]
        targets = [0, 1, 0, 1]

        loss_fn = ErrorRateRegularizationLoss()

        result = loss_fn.forward(
            sensitive_attribute_list=torch.tensor(sensitive_attribute_list),
            device=device,
            predictions=predictions,
            possible_sensitive_attributes=possible_sensitive_attributes,
            possible_targets=possible_targets,
            true_targets=torch.tensor(targets),
            privileged_group=0,
            unprivileged_group=1,
        )

        assert isinstance(result, torch.Tensor)
        assert result.ndim == 0
        assert result >= 0

    def test_calculate_group_error_rate(self):
        loss_fn = ErrorRateRegularizationLoss()
        # FP+TN = 5, TP+FN = 5
        # Error rate = (FP + FN) / (FP + TN + TP + FN) = (1 + 1) / 10 = 0.2
        error_rate = loss_fn._calculate_group_error_rate(
            group=0,
            fp=1.0,
            tn=4.0,
            tp=4.0,
            fn=1.0,
            possible_sensitive_attributes=[0, 1],
            average_probabilities=None,
        )
        assert error_rate == pytest.approx(0.2)

    def test_apply_fairness_mask(self):
        loss_fn = ErrorRateRegularizationLoss()
        fairness_violations = [torch.tensor(0.1), torch.tensor(0.5), torch.tensor(0.2)]
        device = torch.device("cpu")
        masked_violations = loss_fn._apply_fairness_mask(fairness_violations, device)
        # Should return the max value (0.5) because mask is 1 at that index and sum is taken
        assert torch.isclose(masked_violations, torch.tensor(0.5))

    def test_compute_formula_components(self):
        predictions = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        true_targets = torch.tensor([0, 1])
        predictions_argmax = torch.tensor([0, 1])
        sensitive_attribute_list = torch.tensor([0, 1])
        softmax_ = torch.softmax(predictions, dim=1)

        result = ErrorRateRegularizationLoss.compute_formula_components(
            predictions=predictions,
            true_targets=true_targets,
            predictions_argmax=predictions_argmax,
            sensitive_attribute_list=sensitive_attribute_list,
            softmax_=softmax_,
            privileged_group=0,
            unprivileged_group=1,
        )

        # Result is a large tuple (17 elements)
        assert len(result) == 17
        # First 8 are float components (FP, TN, TP, FN for each group)
        # analysis_dict is at index 8
        assert isinstance(result[8], dict)

    def test_violation_with_dataset(self):
        model = nn.Sequential(nn.Linear(2, 2))
        x = torch.randn(10, 2)
        z = torch.randint(0, 2, (10,))
        y = torch.randint(0, 2, (10,))
        dataset = DataLoader(TensorDataset(x, z, y), batch_size=5)

        loss_fn = ErrorRateRegularizationLoss()
        violation = loss_fn.violation_with_dataset(
            model=model,
            dataset=dataset,
            average_probabilities={},
            device=torch.device("cpu"),
            privileged_group=0,
            unprivileged_group=1,
        )

        assert isinstance(violation, torch.Tensor)
        assert violation >= 0
