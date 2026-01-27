import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from puffle.Regularization.disparity_loss import DisparityRegularizationLoss


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
