from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch

from puffle.Regularization.disparity_loss import DisparityRegularizationLoss


class TestDisparityCoverage:
    def test_estimate_violation_with_avg_probs(self):
        loss = DisparityRegularizationLoss()
        # Test _estimate_violation when z!=z is zero
        # known_numerator=10, known_denominator=20 -> 0.5
        # avg_prob for target|0 = 0.3
        # violation = |0.5 - 0.3| = 0.2
        avg_probs = {"0|0": 0.3, "0|1": 0.7}

        # Case: is_z_zero=True (z=0 was the zero group, so we look up avg for z=0? No wait)
        # Helper logic: if is_z_zero: denominator val = 1 (if z=1) else 0.
        # If z=1 and is_z_zero=True (implies z=0 counts are zero?)

        # Let's test the method directly
        val = loss._estimate_violation(
            target=0,
            z=1,
            known_numerator=torch.tensor(10.0),
            known_denominator=torch.tensor(20.0),
            average_probabilities=avg_probs,
            is_z_zero=True,
        )
        # denominator_val = 1 (z=1). prob_key = "0|1" -> 0.7
        # |0.7 - 0.5| = 0.2
        assert val.item() == pytest.approx(0.2)

    def test_update_global_counters_logic(self):
        loss = DisparityRegularizationLoss()
        # Test _update_global_counters logic via forward?
        # Or directly
        global_counters = {}
        json_file = {
            "possible_z": [0, 1],
            "missing_combinations": [("key_to_remove", 0)],
        }
        # Populate counters first
        global_counters["0|0"] = 10
        global_counters["1|0"] = 20
        global_counters["key_to_remove"] = 999

        updated = loss._update_global_counters(global_counters, json_file)
        # Should sum "0|0" and "1|0" into "0" -> 30
        assert updated[0] == 30
        assert "key_to_remove" not in updated

    def test_argmax_zero_cases(self):
        loss = DisparityRegularizationLoss()
        # Test compute_violation_with_argmax edge cases
        # Scenario where one group is missing

        sensitive = torch.tensor([1, 1, 1])  # All 1s
        predictions = torch.tensor([0, 0, 0])
        # Calculation for z=0 (length 0) vs z=1 (length 3)

        res_0 = loss.compute_violation_with_argmax(
            predictions, sensitive, current_target=0, current_sensitive_feature=0
        )
        # z=0 count is 0. z!=0 count is 3.
        # Should return y_eq_k_and_z_not_eq / z_not_eq
        # y=0, z!=0 -> 3/3 = 1.0. abs(1.0) = 1.0
        assert res_0 == 1.0

        # Case both zero (empty input)
        res_empty = loss.compute_violation_with_argmax(
            torch.tensor([]), torch.tensor([]), 0, 0
        )
        assert res_empty == 0

    def test_evaluate_violation_tensor_handling(self):
        loss = DisparityRegularizationLoss()
        # Test evaluate_violation where violation terms are tensors > 1 numel

        with patch.object(loss, "compute_violation_with_argmax") as mock_method:
            # Return a tensor with >1 elements to trigger mean() path?
            # Or just tensor to trigger isinstance check.
            # Code: if item.numel() > 1: append(mean)
            mock_method.return_value = torch.tensor([0.5, 0.6])

            preds = torch.tensor([[0.1, 0.9], [0.9, 0.1]])  # argmax 1, 0
            sens = torch.tensor([0, 1])

            # Need actual tensors for logic to proceed up to call
            # It loops possible_targets * possible_z
            # returns tensor

            res = loss.evaluate_violation(preds, sens, [0], [0])  # 1 call
            # Should be mean of [0.5, 0.6] -> 0.55
            assert res.item() == pytest.approx(0.55)

    def test_compute_violation_term_zero_cases(self):
        loss = DisparityRegularizationLoss()
        # Create scenario:
        # target=0, z=0
        # predictions argmax list: [0, 0, 1]
        # sensitive list:          [0, 1, 1]
        # z=0: index 0 (pred 0) -> z_eq_z=1, y_eq_k_and_z_eq_z=1
        # z!=0 (z=1): indices 1,2 (preds 0, 1) -> z_not_eq_z=2, y_eq_k_and_z_not_eq_z=1
        # This is normal.

        # We need:
        # Case 1: (y_eq_k_and_z_eq_z == 0 and y_eq_k_and_z_not_eq_z != 0)
        # target value 1. sensitive value 0.
        # preds list: [0, 1, 1]
        # sens list:  [0, 1, 1]
        # z=0: pred=0 -> no target 1. -> y_eq_k_and_z_eq_z = 0.
        # z!=0: preds 1,1 -> target 1 found. -> y_eq_k_and_z_not_eq_z = 2.

        softmax_ = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        preds_argmax = torch.tensor([0, 1, 1])
        sens = torch.tensor([0, 1, 1])
        avg_probs = {"1|0": 0.5}  # avg prob for target 1 given z 0

        # Should trigger _estimate_violation(..., is_z_zero=False)
        term, _ = loss._compute_violation_term(
            target=1,
            z=0,
            softmax_=softmax_,
            predictions_argmax=preds_argmax,
            sensitive_attribute_list=sens,
            average_probabilities=avg_probs,
        )
        # _estimate_violation should run. simple coverage check.
        assert term >= 0

        # Case 2: (y_eq_k_and_z_eq_z != 0 and y_eq_k_and_z_not_eq_z == 0)
        # target=0. z=0.
        # preds: [0, 1, 1] -> see above.
        # z=0 -> pred 0. y_eq... = 1.
        # z!=0 -> preds 1,1. y_eq.. (target 0) = 0.
        # Should trigger _estimate_violation (..., is_z_zero=False)
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

        # Test
        res = loss.forward(
            sensitive_attribute_list=[],
            device=torch.device("cpu"),
            predictions=torch.tensor([]),
            possible_sensitive_attributes=[],
            possible_targets=[],
            global_computation=True,
        )
        # Tuple return
        assert isinstance(res, tuple)
        assert res[1] == {"test": 1}
        assert cast("Any", loss._update_global_counters).called
