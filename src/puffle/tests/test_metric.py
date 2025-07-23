import torch
import pytest
import numpy as np
from puffle.FairReg.Utils.metric import compute_demographic_disparity, compute_differentiable_demographic_disparity


class TestMetrics:
    def test_compute_demographic_disparity_basic(self):
        """Test the basic functionality of compute_demographic_disparity."""
        # Create a simple test case
        z = torch.tensor([0, 0, 0, 1, 1, 1])
        y = torch.tensor([0, 0, 1, 1, 1, 1])

        # Expected disparity:
        # P(Y=0|Z=0) = 2/3, P(Y=0|Z=1) = 0/3, diff = 2/3
        # P(Y=1|Z=0) = 1/3, P(Y=1|Z=1) = 3/3, diff = 2/3
        # Max disparity = 2/3
        expected_disparity = 2 / 3

        result = compute_demographic_disparity(z, y)

        assert np.isclose(result, expected_disparity, atol=1e-6)

    def test_compute_demographic_disparity_no_disparity(self):
        """Test when there is no demographic disparity."""
        # Create a test case with no disparity
        z = torch.tensor([0, 0, 1, 1])
        y = torch.tensor([0, 1, 0, 1])

        # Expected disparity:
        # P(Y=0|Z=0) = 1/2, P(Y=0|Z=1) = 1/2, diff = 0
        # P(Y=1|Z=0) = 1/2, P(Y=1|Z=1) = 1/2, diff = 0
        # Max disparity = 0
        expected_disparity = 0.0

        result = compute_demographic_disparity(z, y)

        assert np.isclose(result, expected_disparity, atol=1e-6)

    def test_compute_demographic_disparity_complete_disparity(self):
        """Test when there is complete demographic disparity."""
        # Create a test case with complete disparity
        z = torch.tensor([0, 0, 1, 1])
        y = torch.tensor([0, 0, 1, 1])

        # Expected disparity:
        # P(Y=0|Z=0) = 2/2 = 1, P(Y=0|Z=1) = 0/2 = 0, diff = 1
        # P(Y=1|Z=0) = 0/2 = 0, P(Y=1|Z=1) = 2/2 = 1, diff = 1
        # Max disparity = 1
        expected_disparity = 1.0

        result = compute_demographic_disparity(z, y)

        assert np.isclose(result, expected_disparity, atol=1e-6)

    def test_compute_demographic_disparity_multiple_values(self):
        """Test with multiple values for Z and Y."""
        # Create a test case with multiple values
        z = torch.tensor([0, 0, 1, 1, 2, 2])
        y = torch.tensor([0, 1, 0, 1, 0, 2])

        # Calculate expected disparities manually:
        # For Z=0 vs Z!=0:
        #   P(Y=0|Z=0) = 1/2, P(Y=0|Z!=0) = 2/4 = 0.5, diff = 0
        #   P(Y=1|Z=0) = 1/2, P(Y=1|Z!=0) = 1/4 = 0.25, diff = 0.25
        #   P(Y=2|Z=0) = 0/2 = 0, P(Y=2|Z!=0) = 1/4 = 0.25, diff = 0.25
        # For Z=1 vs Z!=1:
        #   P(Y=0|Z=1) = 1/2, P(Y=0|Z!=1) = 2/4 = 0.5, diff = 0
        #   P(Y=1|Z=1) = 1/2, P(Y=1|Z!=1) = 1/4 = 0.25, diff = 0.25
        #   P(Y=2|Z=1) = 0/2 = 0, P(Y=2|Z!=1) = 1/4 = 0.25, diff = 0.25
        # For Z=2 vs Z!=2:
        #   P(Y=0|Z=2) = 1/2, P(Y=0|Z!=2) = 2/4 = 0.5, diff = 0
        #   P(Y=1|Z=2) = 0/2 = 0, P(Y=1|Z!=2) = 2/4 = 0.5, diff = 0.5
        #   P(Y=2|Z=2) = 1/2, P(Y=2|Z!=2) = 0/4 = 0, diff = 0.5
        # Max disparity = 0.5
        expected_disparity = 0.5

        result = compute_demographic_disparity(z, y)

        assert np.isclose(result, expected_disparity, atol=1e-6)

    def test_compute_demographic_disparity_empty_inputs(self):
        """Test behavior with empty inputs."""
        z = torch.tensor([])
        y = torch.tensor([])

        # Empty tensors should return 0 disparity (no data = no disparity)
        with pytest.raises(ValueError):
            # This should raise a RuntimeError as we'll try to compute mean of empty tensor
            compute_demographic_disparity(z, y)

    def test_compute_demographic_disparity_single_value(self):
        """Test when there is only one value for Z or Y."""
        # Create a test case with a single value for Z
        z = torch.tensor([1, 1, 1])
        y = torch.tensor([0, 1, 0])

        # When all Z values are the same, there is no "Z!=z" group to compare with
        # This should return 0 disparity
        with pytest.raises(ValueError):
            # This should raise a RuntimeError as we can't compute P(Y|Z!=z) when all Z are the same
            result = compute_demographic_disparity(z, y)

    def test_compute_differentiable_demographic_disparity_basic(self):
        """Test the basic functionality of compute_differentiable_demographic_disparity."""
        # Create a simple test case
        predictions_argmax = torch.tensor([0, 0, 1, 1, 1, 1])
        sensitive_attributes = torch.tensor([0, 0, 0, 1, 1, 1])
        # Softmax output for two classes (class 0 and class 1)
        softmax_output = torch.tensor(
            [
                [0.9, 0.1],  # Predicted class 0 with high confidence
                [0.8, 0.2],  # Predicted class 0 with high confidence
                [0.3, 0.7],  # Predicted class 1 with medium confidence
                [0.2, 0.8],  # Predicted class 1 with high confidence
                [0.1, 0.9],  # Predicted class 1 with high confidence
                [0.0, 1.0],  # Predicted class 1 with full confidence
            ]
        )

        # Expected calculation:
        # For target=0, sensitive_attribute=0:
        # Y_eq_k_and_Z_eq_z = 0.9 + 0.8 = 1.7
        # Y_eq_k_and_Z_not_eq_z = 0 (no predictions of class 0 for Z=1)
        # Z_eq_z = 0.9 + 0.8 + 0.3 = 2.0
        # Z_not_eq_z = 0.2 + 0.1 + 0.0 = 0.3
        # violation = |1.7/2.0 - 0/0.3| = |0.85 - 0| = 0.85

        # For target=0, sensitive_attribute=1:
        # Y_eq_k_and_Z_eq_z = 0 (no predictions of class 0 for Z=1)
        # Y_eq_k_and_Z_not_eq_z = 0.9 + 0.8 = 1.7
        # Z_eq_z = 0.2 + 0.1 + 0.0 = 0.3
        # Z_not_eq_z = 0.9 + 0.8 + 0.3 = 2.0
        # violation = |0/0.3 - 1.7/2.0| = |0 - 0.85| = 0.85

        # For target=1, sensitive_attribute=0:
        # Y_eq_k_and_Z_eq_z = 0.7 (only one prediction of class 1 for Z=0)
        # Y_eq_k_and_Z_not_eq_z = 0.8 + 0.9 + 1.0 = 2.7
        # Z_eq_z = 0.1 + 0.2 + 0.7 = 1.0
        # Z_not_eq_z = 0.8 + 0.9 + 1.0 = 2.7
        # violation = |0.7/1.0 - 2.7/2.7| = |0.7 - 1.0| = 0.3

        # For target=1, sensitive_attribute=1:
        # Y_eq_k_and_Z_eq_z = 0.8 + 0.9 + 1.0 = 2.7
        # Y_eq_k_and_Z_not_eq_z = 0.7 (only one prediction of class 1 for Z=0)
        # Z_eq_z = 0.8 + 0.9 + 1.0 = 2.7
        # Z_not_eq_z = 0.1 + 0.2 + 0.7 = 1.0
        # violation = |2.7/2.7 - 0.7/1.0| = |1.0 - 0.7| = 0.3

        # Max violation = 0.85
        expected_disparity = 0.85

        device = torch.device("cpu")
        result = compute_differentiable_demographic_disparity(
            predictions_argmax=predictions_argmax,
            sensitive_attributes=sensitive_attributes,
            softmax_output=softmax_output,
        )

        assert torch.isclose(result, torch.tensor(expected_disparity), atol=1e-6)

    def test_compute_differentiable_demographic_disparity_no_disparity(self):
        """Test when there is no demographic disparity in the differentiable version."""
        # Create a test case with no disparity
        predictions_argmax = torch.tensor([0, 1, 0, 1])
        sensitive_attributes = torch.tensor([0, 0, 1, 1])

        # Softmax output for two classes (class 0 and class 1)
        softmax_output = torch.tensor(
            [
                [0.8, 0.2],  # Predicted class 0 with high confidence
                [0.2, 0.8],  # Predicted class 1 with high confidence
                [0.8, 0.2],  # Predicted class 0 with high confidence
                [0.2, 0.8],  # Predicted class 1 with high confidence
            ]
        )

        # With perfect balance, the disparity should be 0
        expected_disparity = 0.0

        device = torch.device("cpu")
        result = compute_differentiable_demographic_disparity(
            predictions_argmax=predictions_argmax,
            sensitive_attributes=sensitive_attributes,
            softmax_output=softmax_output,
        )

        assert torch.isclose(result, torch.tensor(expected_disparity), atol=1e-6)

    def test_compute_differentiable_demographic_disparity_errors(self):
        """Test error cases for the differentiable version."""
        # Test with empty inputs
        predictions_argmax_empty = torch.tensor([])
        sensitive_attributes_empty = torch.tensor([])
        softmax_output_empty = torch.tensor([])

        with pytest.raises(ValueError):
            compute_differentiable_demographic_disparity(
                predictions_argmax=predictions_argmax_empty,
                sensitive_attributes=sensitive_attributes_empty,
                softmax_output=softmax_output_empty,
            )

        # Test with single value for sensitive attribute
        predictions_argmax_single = torch.tensor([0, 1, 0])
        sensitive_attributes_single = torch.tensor([1, 1, 1])
        softmax_output_single = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3]])

        with pytest.raises(ValueError):
            compute_differentiable_demographic_disparity(
                predictions_argmax=predictions_argmax_single,
                sensitive_attributes=sensitive_attributes_single,
                softmax_output=softmax_output_single,
            )
