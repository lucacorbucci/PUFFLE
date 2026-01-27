from typing import Any, cast

import pytest
import torch

from puffle.Utils.metric import (
    compute_demographic_disparity,
    compute_differentiable_demographic_disparity,
)


class TestMetricCoverage:
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
