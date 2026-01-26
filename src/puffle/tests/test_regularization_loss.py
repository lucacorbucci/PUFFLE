
import numpy as np
import torch
from FairReg.RegularizationLoss import RegularizationLoss


class TestRegularization:
    def test_compute_probabilities_output_type(self):
        # Test if the output is a tuple containing two dictionaries
        predictions = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        sensitive_attribute_list = [0, 1]
        device = torch.device("cpu")
        possible_sensitive_attributes = [0, 1]
        possible_targets = [0, 1]
        binary_sensitive_value = True
        result = RegularizationLoss.compute_probabilities(
            predictions,
            sensitive_attribute_list,
            device,
            possible_sensitive_attributes,
            possible_targets,
            binary_sensitive_value,
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
        binary_sensitive_value = True
        probabilities, _ = RegularizationLoss.compute_probabilities(
            predictions,
            sensitive_attribute_list,
            device,
            possible_sensitive_attributes,
            possible_targets,
            binary_sensitive_value,
        )

        for target in possible_targets:
            for z in possible_sensitive_attributes:
                key = f"{target}|{z}"
                assert key in probabilities
                assert 0 <= probabilities[key] <= 1

    def test_compute_probabilities_counters(self):
        # Test if counters are non-negative
        predictions = torch.tensor([[0.7, 0.3], [0.4, 0.6]])
        sensitive_attribute_list = [0, 1]
        device = torch.device("cpu")
        possible_sensitive_attributes = [0, 1]
        possible_targets = [0, 1]
        binary_sensitive_value = True
        _, counters = RegularizationLoss.compute_probabilities(
            predictions,
            sensitive_attribute_list,
            device,
            possible_sensitive_attributes,
            possible_targets,
            binary_sensitive_value,
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
        binary_sensitive_value = True

        expected_counters = {
            "0|0": 4,
            "0|1": 1,
            "1|0": 1,
            "1|1": 4,
            "0": 5,
            "1": 5,
        }

        # Compute actual outputs
        _, actual_counters = RegularizationLoss.compute_probabilities(
            predictions,
            sensitive_attribute_list,
            device,
            possible_sensitive_attributes,
            possible_targets,
            binary_sensitive_value,
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
        binary_sensitive_value = True

        expected_probabilities = {
            "0|0": torch.tensor(2.3883),
            "0": torch.tensor(5),
            "0|1": torch.tensor(0.5498),
            "1": torch.tensor(5),
            "1|0": torch.tensor(0.5987),
            "1|1": torch.tensor(2.2952),
        }

        # Compute actual outputs
        actual_probabilities, _ = RegularizationLoss.compute_probabilities(
            predictions,
            sensitive_attribute_list,
            device,
            possible_sensitive_attributes,
            possible_targets,
            binary_sensitive_value,
        )

        # Compare expected and actual outputs

        for key, value in actual_probabilities.items():
            current_value = torch.tensor(value) if isinstance(value, int) else value
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

        expected_result = 0  # Expected DPL value for this example

        # Calculate the actual result
        actual_result = RegularizationLoss().compute_violation_with_argmax(
            predictions_argmax,
            sensitive_attribute_list,
            current_target,
            current_sensitive_feature,
        )

        # Compare the expected and actual results
        assert actual_result == expected_result

    def test_compute_violation_with_argmax_z_eq_z_argmax_zero(self):
        # Test the compute_violation_with_argmax method when Z_eq_z_argmax == 0
        predictions_argmax = torch.tensor([0, 1, 0, 1, 0, 0, 1, 1])
        sensitive_attribute_list = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
        current_target = 1
        current_sensitive_feature = 0

        expected_result = 0.5  # Expected DPL value when Z_eq_z_argmax is zero

        # Calculate the actual result
        actual_result = RegularizationLoss().compute_violation_with_argmax(
            predictions_argmax,
            sensitive_attribute_list,
            current_target,
            current_sensitive_feature,
        )

        # Compare the expected and actual results
        assert np.isclose(actual_result, expected_result, atol=1e-6)  # Use np.isclose for floating-point comparisons

    def test_compute_violation_with_argmax_z_not_eq_z_argmax_zero(self):
        # Test the compute_violation_with_argmax method when Z_not_eq_z_argmax == 0
        predictions_argmax = torch.tensor([0, 1, 0, 1, 0, 0, 1, 1])
        sensitive_attribute_list = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1])
        current_target = 1
        current_sensitive_feature = 0

        expected_result = 0.5  # Expected DPL value when Z_not_eq_z_argmax is zero

        # Calculate the actual result
        actual_result = RegularizationLoss().compute_violation_with_argmax(
            predictions_argmax,
            sensitive_attribute_list,
            current_target,
            current_sensitive_feature,
        )

        # Compare the expected and actual results
        assert np.isclose(actual_result, expected_result, atol=1e-6)  # Use np.isclose for floating-point comparisons

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
        binary_sensitive_value = True

        regularization_loss = RegularizationLoss()

        result = regularization_loss.forward(
            sensitive_attribute_list=sensitive_attribute_list,
            device=device,
            predictions=predictions,
            possible_sensitive_attributes=possible_sensitive_attributes,
            possible_targets=possible_targets,
            binary_sensitive_value=binary_sensitive_value,
        )

        expected_result = torch.tensor(0.3677)

        assert np.isclose(result, expected_result, atol=1e-4)
