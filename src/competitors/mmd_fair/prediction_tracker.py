# ABOUTME: Server-side prediction tracking for MMD-Fair FL.
# ABOUTME: Port of Fair-FL's YSet class for maintaining rolling windows of predictions.

import numpy as np
import torch


class PredictionTracker:
    """
    Maintains a rolling window of predictions for a demographic group.

    Port of Fair-FL's YSet class. Stores predictions and supports
    drop-and-refresh protocol from Algorithm 2.
    """

    def __init__(self, demographic_group: int, capacity: int, device: str = "cpu"):
        """
        Initialize prediction tracker.

        Args:
            demographic_group: Demographic group ID (0 or 1)
            capacity: Maximum number of predictions to store
            device: PyTorch device

        """
        self.demographic_group = demographic_group
        self.capacity = capacity
        self.device = device
        self.predictions = torch.zeros(capacity, device=device)

    def drop(self, mu: float) -> None:
        """
        Randomly drop mu fraction of predictions.

        Implements the drop step from Fair-FL Algorithm 2.

        Args:
            mu: Fraction of predictions to drop (0.0 to 1.0)

        """
        current_size = len(self.predictions)
        keep_size = int((1 - mu) * current_size)

        if keep_size == 0:
            self.predictions = torch.zeros(0, device=self.device)
            return

        # Randomly select indices to keep
        keep_indices = np.random.choice(current_size, size=keep_size, replace=False)
        self.predictions = self.predictions[keep_indices]

    def update(self, new_predictions: list[torch.Tensor]) -> None:
        """
        Add new predictions to the tracker.

        Args:
            new_predictions: List of prediction tensors from clients

        """
        # Flatten and concatenate all new predictions
        updates = torch.cat([pred.flatten() for pred in new_predictions], dim=0)

        # Move to tracker's device if needed
        if updates.device != torch.device(self.device):
            updates = updates.to(self.device)

        # Concatenate with existing predictions
        self.predictions = torch.cat((self.predictions, updates), dim=0)

    def get_predictions(self) -> torch.Tensor:
        """Return current predictions tensor."""
        return self.predictions

    def __len__(self) -> int:
        """Return number of predictions currently stored."""
        return len(self.predictions)
