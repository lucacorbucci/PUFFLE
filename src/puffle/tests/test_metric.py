from typing import Any, cast

import numpy as np
import pytest
import torch

from puffle.Utils.metric import (
    compute_demographic_disparity,
    compute_differentiable_demographic_disparity,
)


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

        result, _ = compute_demographic_disparity(z, y)

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

        result, _ = compute_demographic_disparity(z, y)

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

        result, _ = compute_demographic_disparity(z, y)

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

        result, _ = compute_demographic_disparity(z, y)

        assert np.isclose(result, expected_disparity, atol=1e-6)

    def test_compute_demographic_disparity_empty_inputs(self):
        """Test behavior with empty inputs."""
        z = torch.tensor([])
        y = torch.tensor([])

        # Empty tensors should raise ValueError based on implementation checks
        with pytest.raises(ValueError):
            compute_demographic_disparity(z, y)

    def test_compute_demographic_disparity_single_value(self):
        """Test when there is only one value for Z or Y."""
        # Create a test case with a single value for Z
        z = torch.tensor([1, 1, 1])
        y = torch.tensor([0, 1, 0])

        # When all Z values are the same, it might raise ValueError or return 0
        # The current implementation might raise ValueError or handle it.
        # Let's check the code: it does mean().item() on (y[z != z_val] == y_val)
        # if z != z_val is empty, mean() of empty tensor is NaN.
        # When all Z values are the same, it might raise ValueError or return 0
        # The current implementation might raise ValueError or handle it.
        # In the existing test it was raises(ValueError). Let's keep it.

        # In the existing test it was raises(ValueError). Let's keep it.
        with pytest.raises(ValueError):
            compute_demographic_disparity(z, y)

    def test_compute_differentiable_demographic_disparity_basic(self):
        """Test the basic functionality of compute_differentiable_demographic_disparity."""
        predictions_argmax = torch.tensor([0, 0, 1, 1, 1, 1])
        sensitive_attributes = torch.tensor([0, 0, 0, 1, 1, 1])
        softmax_output = torch.tensor(
            [
                [0.9, 0.1],
                [0.8, 0.2],
                [0.3, 0.7],
                [0.2, 0.8],
                [0.1, 0.9],
                [0.0, 1.0],
            ]
        )

        expected_disparity = 0.85

        result = compute_differentiable_demographic_disparity(
            predictions_argmax=predictions_argmax,
            sensitive_attributes=sensitive_attributes,
            softmax_output=softmax_output,
        )

        assert torch.isclose(result, torch.tensor(expected_disparity), atol=1e-6)

    def test_compute_differentiable_demographic_disparity_no_disparity(self):
        """Test when there is no demographic disparity."""
        predictions_argmax = torch.tensor([0, 1, 0, 1])
        sensitive_attributes = torch.tensor([0, 0, 1, 1])

        softmax_output = torch.tensor(
            [
                [0.8, 0.2],
                [0.2, 0.8],
                [0.8, 0.2],
                [0.2, 0.8],
            ]
        )

        expected_disparity = 0.0

        result = compute_differentiable_demographic_disparity(
            predictions_argmax=predictions_argmax,
            sensitive_attributes=sensitive_attributes,
            softmax_output=softmax_output,
        )

        assert torch.isclose(result, torch.tensor(expected_disparity), atol=1e-6)

    def test_compute_differentiable_demographic_disparity_errors(self):
        """Test error cases for the differentiable version."""
        predictions_argmax_empty = torch.tensor([], dtype=torch.long)
        sensitive_attributes_empty = torch.tensor([], dtype=torch.long)
        softmax_output_empty = torch.tensor([])

        with pytest.raises(ValueError):
            compute_differentiable_demographic_disparity(
                predictions_argmax=predictions_argmax_empty,
                sensitive_attributes=sensitive_attributes_empty,
                softmax_output=softmax_output_empty,
            )

        predictions_argmax_single = torch.tensor([0, 1, 0])
        sensitive_attributes_single = torch.tensor([1, 1, 1])
        softmax_output_single = torch.tensor([[0.8, 0.2], [0.2, 0.8], [0.7, 0.3]])

        with pytest.raises(ValueError):
            compute_differentiable_demographic_disparity(
                predictions_argmax=predictions_argmax_single,
                sensitive_attributes=sensitive_attributes_single,
                softmax_output=softmax_output_single,
            )

    def test_demographic_disparity_errors(self):
        # Length mismatch
        with pytest.raises(ValueError, match="same length"):
            compute_demographic_disparity(torch.zeros(5), torch.zeros(4))

        # Wrong type
        with pytest.raises(TypeError, match=r"torch\.Tensor"):
            compute_demographic_disparity(cast("Any", [0, 1]), torch.zeros(2))

        # Empty input (covered but good to ensure)
        with pytest.raises(ValueError, match="not be empty"):
            compute_demographic_disparity(torch.tensor([]), torch.tensor([]))

    def test_differentiable_disparity_errors(self):
        # Length mismatch
        with pytest.raises(ValueError, match="same length"):
            compute_differentiable_demographic_disparity(
                torch.zeros(5), torch.zeros(4), torch.zeros((5, 2))
            )

        # Wrong type
        with pytest.raises(TypeError, match=r"torch\.Tensor"):
            compute_differentiable_demographic_disparity(
                cast("Any", [0, 1]), torch.zeros(2), torch.zeros((2, 2))
            )

        # Empty input
        with pytest.raises(ValueError, match="not be empty"):
            compute_differentiable_demographic_disparity(
                torch.tensor([]), torch.tensor([]), torch.tensor([])
            )

        # Single value
        with pytest.raises(ValueError, match="more than one unique value"):
            compute_differentiable_demographic_disparity(
                torch.zeros(5), torch.zeros(5), torch.zeros((5, 2))
            )

    def test_demographic_disparity_single_attribute(self):
        # Metric requires >1 sensitive attribute
        with pytest.raises(ValueError, match="At least two"):
            compute_demographic_disparity(torch.zeros(5), torch.zeros(5))
