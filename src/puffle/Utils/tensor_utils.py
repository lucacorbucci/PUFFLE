# ABOUTME: Provides utility functions for tensor handling and conversion.
# ABOUTME: Reduces code duplication for common tensor operations.

import torch

from puffle.Utils.constants import EPSILON


def ensure_tensor(
    data: torch.Tensor | list,
    device: torch.device | str,
    dtype: torch.dtype = torch.long,
) -> torch.Tensor:
    """
    Convert data to tensor on the specified device if needed.

    If data is already a tensor on the correct device, returns it directly
    to avoid unnecessary allocations.
    """
    if isinstance(data, torch.Tensor):
        if data.device == torch.device(device) and data.dtype == dtype:
            return data
        return data.to(device=device, dtype=dtype)
    return torch.tensor(data, device=device, dtype=dtype)


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """Perform division with numerical stability."""
    return numerator / (denominator + EPSILON)
