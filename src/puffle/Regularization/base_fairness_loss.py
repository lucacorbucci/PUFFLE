from abc import abstractmethod

import torch
from torch import nn

from puffle.Utils.constants import DEFAULT_ESTIMATION
from puffle.Utils.tensor_utils import ensure_tensor


class BaseFairnessLoss(nn.Module):
    """Abstract base class for fairness regularization losses."""

    def __init__(self, *, estimation: float = DEFAULT_ESTIMATION) -> None:
        super().__init__()
        self.estimation = estimation

    @abstractmethod
    def forward(
        self,
        sensitive_attribute_list: torch.Tensor,
        device: torch.device,
        predictions: torch.Tensor,
        possible_sensitive_attributes: list,
        possible_targets: list,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Compute the regularization loss."""

    def _apply_fairness_mask(
        self,
        fairness_violations: list[torch.Tensor],
        device: torch.device | str,
    ) -> torch.Tensor:
        """
        Apply max violation mask for gradient routing.

        Routes gradients only through the maximum violation term to avoid
        averaging out signal from the most significant violation.
        """
        if not fairness_violations:
            return torch.tensor(0.0, device=device)

        # Convert list to tensor and find max
        # We need to handle both tensors and floats in the list
        fairness_violations_tensors = []
        for v in fairness_violations:
            if isinstance(v, torch.Tensor):
                # If a violation is a multi-element tensor (e.g. per-sample violations),
                # we reduce it to a scalar before stacking
                if v.numel() > 1:
                    fairness_violations_tensors.append(v.mean())
                else:
                    fairness_violations_tensors.append(v.view(()))  # Ensure scalar
            else:
                fairness_violations_tensors.append(
                    torch.tensor(float(v), device=device)
                )

        fairness_stack = torch.stack(fairness_violations_tensors)

        # We use .detach() for finding the index to avoid carrying gradients through index selection
        # Note: argmax on a multi-dimensional tensor would need reduction,
        # but here we've ensured they are all scalars.
        max_idx = torch.argmax(fairness_stack.detach())

        mask = torch.zeros_like(fairness_stack)
        mask[max_idx] = 1.0

        return (fairness_stack * mask).sum()

    def _prepare_sensitive_attributes(
        self,
        sensitive_attribute_list: torch.Tensor | list,
        device: torch.device | str,
    ) -> torch.Tensor:
        """Convert sensitive attributes to tensor on device."""
        return ensure_tensor(sensitive_attribute_list, device, torch.long)
