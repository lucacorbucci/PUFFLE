# ABOUTME: Defines type aliases for complex types used across the codebase.
# ABOUTME: Improves readability and consistency of type annotations.

from typing import TypeAlias

import torch

# Input types
SensitiveAttributes: TypeAlias = torch.Tensor | list[int]
Predictions: TypeAlias = torch.Tensor
Targets: TypeAlias = torch.Tensor

# Output types
# MetricsDict represents a dictionary of metric names mapped to a list of values over time
MetricsDict: TypeAlias = dict[str, list[float]]
# EpochMetrics represents a dictionary of metric names mapped to their current value
EpochMetrics: TypeAlias = dict[str, float]

# Configuration types
DeviceType: TypeAlias = torch.device | str
