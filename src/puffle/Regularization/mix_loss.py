"""
Combined loss function for fairness-aware training.

This module provides MixLoss, which blends standard task loss (e.g., cross-entropy)
with a fairness regularization term based on a tunable lambda parameter.
"""

from torch import nn


class MixLoss(nn.Module):
    """
    Combined loss balancing task accuracy and fairness.

    MixLoss computes: (1 - λ) * model_loss + λ * unfairness_loss
    where λ controls the trade-off between task performance and fairness.
    """

    def __init__(
        self,
        model_loss,
        unfairness_loss,
        *,
        possible_sensitive_attributes=None,
        possible_targets=None,
        reduction="mean",
        device="cpu",
    ):
        """
        Initialize mixed loss.

        Args:
            model_loss: Primary task loss (e.g., CrossEntropyLoss).
            unfairness_loss: Fairness regularization loss (e.g., DisparityRegularizationLoss).
            possible_sensitive_attributes: List of valid sensitive attribute values.
            possible_targets: List of valid target values.
            reduction: Reduction method for loss aggregation.
            device: Device for computation.

        """
        super().__init__()
        self.model_criterion = model_loss
        self.unfairness_criterion = unfairness_loss
        self.reduction = reduction
        self.possible_sensitive_attributes = possible_sensitive_attributes or [0, 1]
        self.possible_targets = possible_targets or [0, 1]
        self.device = device

    def forward(self, inputs, target):
        """
        Compute combined loss.

        Args:
            inputs: Tuple of (model_output, sensitive_attribute, lambda_regularization).
            target: Ground truth labels.

        Returns:
            torch.Tensor: Combined loss value.

        """
        model_output = inputs[0]
        sensitive_value = inputs[1]
        lambda_regularization = inputs[2]

        model_loss = self.model_criterion(model_output, target)
        unfairness_loss = self.unfairness_criterion(
            sensitive_attribute_list=sensitive_value,
            device=self.device,
            predictions=model_output,
            possible_sensitive_attributes=self.possible_sensitive_attributes,
            possible_targets=self.possible_targets,
        )

        return (
            1 - lambda_regularization
        ) * model_loss + lambda_regularization * unfairness_loss
