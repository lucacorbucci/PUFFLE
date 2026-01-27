from typing import Any, cast
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from puffle.Regularization.base_fairness_loss import BaseFairnessLoss
from puffle.Regularization.disparity_loss import DisparityRegularizationLoss


class ConcreteFairnessLoss(BaseFairnessLoss):
    def forward(self, *_args, **_kwargs):
        return torch.tensor(0.0)


class TestDisparityLoss:
    def test_compute_probabilities_output_type(self):
        # Test if the output is a tuple containing two dictionaries
        predictions = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        sensitive_attribute_list = [0, 1]
        device = torch.device("cpu")
        possible_sensitive_attributes = [0, 1]
        possible_targets = [0, 1]

        result = DisparityRegularizationLoss.compute_probabilities(
            predictions,
            sensitive_attribute_list,
            device,
            possible_sensitive_attributes,
            possible_targets,
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(item, dict) for item in result)

    def test_compute_probabilities_probabilities_sum(self):
        # Test if the probabilities sum up to 1 for each (target, sensitive) pair
        predictions = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        sensitive_attribute_list = [0, 1]
        device = torch.device("cpu")
        possible_sensitive_attributes = [0, 1]
        possible_targets = [0, 1]

        probabilities, _ = DisparityRegularizationLoss.compute_probabilities(
            predictions,
            sensitive_attribute_list,
            device,
            possible_sensitive_attributes,
            possible_targets,
        )

        for target in possible_targets:
            for z in possible_sensitive_attributes:
                key = f"{target}|{z}"
                assert key in probabilities
                # Note: These are softmax results, so they don't necessarily sum to 1 over (target, z)
                # but they should be between 0 and 1
                assert 0 <= probabilities[key] <= 1

    def test_compute_probabilities_counters(self):
        # Test if counters are non-negative
        predictions = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        sensitive_attribute_list = [0, 1]
        device = torch.device("cpu")
        possible_sensitive_attributes = [0, 1]
        possible_targets = [0, 1]

        _, counters = DisparityRegularizationLoss.compute_probabilities(
            predictions,
            sensitive_attribute_list,
            device,
            possible_sensitive_attributes,
            possible_targets,
        )

        for target in possible_targets:
            for z in possible_sensitive_attributes:
                key = f"{target}|{z}"
                assert key in counters
                assert counters[key] >= 0
                key = f"{z}"
                assert key in counters
                assert counters[key] >= 0

    def test_compute_probabilities_should_return_correct_counters(self):
        predictions = torch.tensor(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.4, 0.6],
                [0.6, 0.4],
                [0.4, 0.6],
                [0.7, 0.3],
                [0.6, 0.4],
                [0.6, 0.4],
                [0.4, 0.6],
            ]
        )
        sensitive_attribute_list = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        device = torch.device("cpu")
        possible_sensitive_attributes = [0, 1]
        possible_targets = [0, 1]

        expected_counters = {
            "0|0": 4,
            "0|1": 1,
            "1|0": 1,
            "1|1": 4,
            "0": 5,
            "1": 5,
        }

        # Compute actual outputs
        _, actual_counters = DisparityRegularizationLoss.compute_probabilities(
            predictions,
            sensitive_attribute_list,
            device,
            possible_sensitive_attributes,
            possible_targets,
        )

        # Compare expected and actual outputs
        assert actual_counters == expected_counters

    def test_compute_probabilities_should_return_correct_probabilities(self):
        predictions = torch.tensor(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.4, 0.6],
                [0.6, 0.4],
                [0.4, 0.6],
                [0.7, 0.3],
                [0.6, 0.4],
                [0.6, 0.4],
                [0.4, 0.6],
            ]
        )

        sensitive_attribute_list = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        device = torch.device("cpu")
        possible_sensitive_attributes = [0, 1]
        possible_targets = [0, 1]

        expected_probabilities = {
            "0|0": torch.tensor(2.3883),
            "0": torch.tensor(5),
            "0|1": torch.tensor(0.5498),
            "1": torch.tensor(5),
            "1|0": torch.tensor(0.5987),
            "1|1": torch.tensor(2.2952),
        }

        # Compute actual outputs
        actual_probabilities, _ = DisparityRegularizationLoss.compute_probabilities(
            predictions,
            sensitive_attribute_list,
            device,
            possible_sensitive_attributes,
            possible_targets,
        )

        # Compare expected and actual outputs
        for key, value in actual_probabilities.items():
            current_value = (
                torch.tensor(value) if isinstance(value, (int, float)) else value
            )
            assert torch.isclose(
                expected_probabilities[key],
                current_value,
                atol=1e-4,
            )

    def test_compute_violation_with_argmax(self):
        # Test the compute_violation_with_argmax method
        predictions_argmax = torch.tensor([0, 1, 0, 1, 0, 0, 1, 1])
        sensitive_attribute_list = torch.tensor([0, 1, 1, 0, 0, 1, 1, 0])
        current_target = 1
        current_sensitive_feature = 1

        expected_result = 0  # Expected disparity value for this balanced case

        # Calculate the actual result
        actual_result = DisparityRegularizationLoss().compute_violation_with_argmax(
            predictions_argmax,
            sensitive_attribute_list,
            current_target,
            current_sensitive_feature,
        )

        # Compare the expected and actual results
        assert actual_result == expected_result

    def test_compute_violation_with_argmax_unbalanced(self):
        # Test with unbalanced case
        predictions_argmax = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
        sensitive_attribute_list = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0])
        current_target = 1
        current_sensitive_feature = 1

        # P(Y=1|Z=1) = 4/4 = 1.0
        # P(Y=1|Z=0) = 0/4 = 0.0
        # |1.0 - 0.0| = 1.0
        expected_result = 1.0

        actual_result = DisparityRegularizationLoss().compute_violation_with_argmax(
            predictions_argmax,
            sensitive_attribute_list,
            current_target,
            current_sensitive_feature,
        )

        assert np.isclose(actual_result, expected_result, atol=1e-6)

    def test_forward_with_binary_sensitive_value(self):
        predictions = torch.tensor(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.4, 0.6],
                [0.6, 0.4],
                [0.4, 0.6],
                [0.7, 0.3],
                [0.6, 0.4],
                [0.6, 0.4],
                [0.4, 0.6],
            ]
        )

        sensitive_attribute_list = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        device = torch.device("cpu")
        possible_sensitive_attributes = [0, 1]
        possible_targets = [0, 1]

        disparity_loss = DisparityRegularizationLoss()

        result = disparity_loss.forward(
            sensitive_attribute_list=sensitive_attribute_list,
            device=device,
            predictions=predictions,
            possible_sensitive_attributes=possible_sensitive_attributes,
            possible_targets=possible_targets,
        )

        # Expected result from original tests was 0.3677
        expected_result = torch.tensor(0.3677)

        assert isinstance(result, torch.Tensor)
        assert torch.isclose(result, expected_result, atol=1e-4)

    def test_apply_fairness_mask(self):
        loss_fn = DisparityRegularizationLoss()
        fairness_violations = [torch.tensor(0.1), torch.tensor(0.5), torch.tensor(0.2)]
        device = torch.device("cpu")
        masked_violations = loss_fn._apply_fairness_mask(fairness_violations, device)
        # 0.5 is the max, so it should be returned
        assert torch.isclose(masked_violations, torch.tensor(0.5))

    def test_violation_with_dataset(self):
        model = nn.Sequential(nn.Linear(2, 2))
        x = torch.randn(10, 2)
        z = torch.randint(0, 2, (10,))
        y = torch.randint(0, 2, (10,))
        dataset = DataLoader(TensorDataset(x, z, y), batch_size=5)

        loss_fn = DisparityRegularizationLoss()
        violation = loss_fn.violation_with_dataset(
            model=model,
            dataset=dataset,
            average_probabilities={},
            device=torch.device("cpu"),
        )
        assert isinstance(violation, torch.Tensor)
        assert violation >= 0

    def test_evaluate_violation(self):
        predictions_argmax = torch.tensor([0, 1, 0, 1])
        sensitive_attribute_list = torch.tensor([0, 0, 1, 1])
        loss_fn = DisparityRegularizationLoss()
        violation = loss_fn.evaluate_violation(
            predictions_argmax=predictions_argmax,
            sensitive_attribute_list=sensitive_attribute_list,
            possible_sensitive_attributes=[0, 1],
            possible_targets=[0, 1],
        )
        assert isinstance(violation, torch.Tensor)
        assert violation >= 0

    def test_estimate_violation_with_avg_probs(self):
        loss = DisparityRegularizationLoss()
        avg_probs = {"0|0": 0.3, "0|1": 0.7}

        val = loss._estimate_violation(
            target=0,
            z=1,
            known_numerator=torch.tensor(10.0),
            known_denominator=torch.tensor(20.0),
            average_probabilities=avg_probs,
            is_z_zero=True,
        )
        assert val.item() == pytest.approx(0.2)

    def test_update_global_counters_logic(self):
        loss = DisparityRegularizationLoss()
        global_counters = {}
        json_file = {
            "possible_z": [0, 1],
            "missing_combinations": [("key_to_remove", 0)],
        }
        global_counters["0|0"] = 10
        global_counters["1|0"] = 20
        global_counters["key_to_remove"] = 999

        updated = loss._update_global_counters(global_counters, json_file)
        assert updated[0] == 30
        assert "key_to_remove" not in updated

    def test_argmax_zero_cases(self):
        loss = DisparityRegularizationLoss()
        sensitive = torch.tensor([1, 1, 1])
        predictions = torch.tensor([0, 0, 0])

        res_0 = loss.compute_violation_with_argmax(
            predictions, sensitive, current_target=0, current_sensitive_feature=0
        )
        assert res_0 == 1.0

        res_empty = loss.compute_violation_with_argmax(
            torch.tensor([]), torch.tensor([]), 0, 0
        )
        assert res_empty == 0

    def test_evaluate_violation_tensor_handling(self):
        loss = DisparityRegularizationLoss()
        with patch.object(loss, "compute_violation_with_argmax") as mock_method:
            mock_method.return_value = torch.tensor([0.5, 0.6])
            preds = torch.tensor([[0.1, 0.9], [0.9, 0.1]])
            sens = torch.tensor([0, 1])
            res = loss.evaluate_violation(preds, sens, [0], [0])
            assert res.item() == pytest.approx(0.55)

    def test_compute_violation_term_zero_cases(self):
        loss = DisparityRegularizationLoss()
        softmax_ = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        preds_argmax = torch.tensor([0, 1, 1])
        sens = torch.tensor([0, 1, 1])
        avg_probs = {"1|0": 0.5}

        term, _ = loss._compute_violation_term(
            target=1,
            z=0,
            softmax_=softmax_,
            predictions_argmax=preds_argmax,
            sensitive_attribute_list=sens,
            average_probabilities=avg_probs,
        )
        assert term >= 0

        term2, _ = loss._compute_violation_term(
            target=0,
            z=0,
            softmax_=softmax_,
            predictions_argmax=preds_argmax,
            sensitive_attribute_list=sens,
            average_probabilities=avg_probs,
        )
        assert term2 >= 0

    def test_forward_global_computation(self):
        loss = DisparityRegularizationLoss()
        loss._prepare_data = MagicMock(
            return_value=(
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                [0],
                [0],
            )
        )
        loss._compute_violation_term = MagicMock(return_value=(torch.tensor(0.5), 10))
        loss._apply_fairness_mask = MagicMock(return_value=torch.tensor(0.5))
        loss._update_global_counters = MagicMock(return_value={"test": 1})

        res = loss.forward(
            sensitive_attribute_list=[],
            device=torch.device("cpu"),
            predictions=torch.tensor([]),
            possible_sensitive_attributes=[],
            possible_targets=[],
            global_computation=True,
        )
        assert isinstance(res, tuple)
        assert res[1] == {"test": 1}
        assert cast("Any", loss._update_global_counters).called

    def test_base_fairness_loss_empty_violations(self):
        loss = ConcreteFairnessLoss()
        res = loss._apply_fairness_mask([], "cpu")
        assert res.item() == 0.0
        assert res.device.type == "cpu"

    def test_estimate_violation_not_z_zero(self):
        loss = DisparityRegularizationLoss()
        avg_probs = {"1|0": 0.3}
        res = loss._estimate_violation(
            target=1,
            z=1,
            known_numerator=0.5,
            known_denominator=1.0,
            average_probabilities=avg_probs,
            is_z_zero=False,
        )
        assert res == pytest.approx(0.2)

    def test_update_global_counters_no_json(self):
        loss = DisparityRegularizationLoss()
        counters = {"a": 1}
        res = loss._update_global_counters(counters, None)
        assert res == counters

    def test_update_global_counters_exception(self):
        loss = DisparityRegularizationLoss()
        json_file = {"possible_z": [[1, 2]]}
        global_counters = {}
        res = loss._update_global_counters(global_counters, json_file)
        assert res == {}

    def test_compute_violation_with_argmax_single_group(self):
        loss = DisparityRegularizationLoss()
        predictions_argmax = torch.tensor([1, 1, 1])
        sensitive_attribute_list = torch.tensor([0, 0, 0])
        res = loss.compute_violation_with_argmax(
            predictions_argmax, sensitive_attribute_list, 1, 0
        )
        assert res == 1.0

    def test_evaluate_violation_scalar_tensors(self):
        loss = DisparityRegularizationLoss()

        def mock_compute(*_args, **_kwargs):
            return torch.tensor(0.5)

        loss.compute_violation_with_argmax = mock_compute  # type: ignore[unresolved-attribute]
        predictions_argmax = torch.tensor([1, 0])
        sensitive_attribute_list = torch.tensor([1, 0])
        res = loss.evaluate_violation(
            predictions_argmax, sensitive_attribute_list, [0, 1], [0, 1]
        )
        assert res.item() == 0.5
