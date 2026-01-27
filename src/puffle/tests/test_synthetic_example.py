# ABOUTME: Integration test for the synthetic dataset example.
# ABOUTME: Verifies that the Puffle model reduces disparity compared to a standard model.

import random

import numpy as np
import pytest
import torch

from puffle.examples.synthetic.main import train_example


class TestSyntheticIntegration:
    def test_synthetic_fairness_improvement(self):
        """
        Run the full synthetic example and verify that the fair model
        achieves lower disparity than the standard model.
        """
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)  # noqa: NPY002
        random.seed(42)

        std_metrics, fair_metrics = train_example()

        final_disp_std = std_metrics["val_disparity"][-1]
        final_disp_fair = fair_metrics["val_disparity"][-1]

        # We expect the fair model to reduce disparity significantly (target ~0.1)
        # Allow some margin for stochasticity
        assert final_disp_fair <= 0.2
        assert final_disp_fair < final_disp_std

        # Check that we maintained reasonable accuracy
        final_acc_std = std_metrics["val_accuracy"][-1]
        final_acc_fair = fair_metrics["val_accuracy"][-1]

        # Standard ~0.82, Fair ~0.70
        assert final_acc_fair >= final_acc_std - 0.20


if __name__ == "__main__":
    pytest.main(["-v", __file__])
