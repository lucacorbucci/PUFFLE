# ABOUTME: Model architectures matching Fair-FL reference implementation.
# ABOUTME: Used for reproducing Fair-FL paper experiments with PUFFLE's MMD-Fair.

import torch
from torch import nn


class TwoLayerNN(nn.Module):
    """
    Two-layer neural network matching Fair-FL's architecture.

    Used for COMPAS and Communities & Crime datasets in Fair-FL experiments.
    Architecture: Linear(input_size, 16) -> ReLU -> Linear(16, 1)
    """

    def __init__(self, input_size: int, hidden_size: int = 16, output_size: int = 1):
        """
        Initialize two-layer neural network.

        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size (default: 16, matching Fair-FL)
            output_size: Output size (default: 1 for binary classification)
        """
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.linear2 = nn.Linear(hidden_size, output_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output logits of shape (batch_size, output_size)
        """
        x = torch.nn.functional.relu(self.linear1(x))
        return self.linear2(x)


class LogisticModel(nn.Module):
    """
    Logistic regression model matching Fair-FL's architecture.

    Used for Synthetic dataset in Fair-FL experiments.
    Architecture: Linear(input_size, 1)
    """

    def __init__(self, input_size: int, output_size: int = 1):
        """
        Initialize logistic regression model.

        Args:
            input_size: Number of input features
            output_size: Output size (default: 1 for binary classification)
        """
        super().__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output logits of shape (batch_size, output_size)
        """
        return self.linear(x)
