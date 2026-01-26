from abc import ABC, abstractmethod

import torch
from torch import nn


class Model(ABC):
    """
    Abstract base class for PyTorch models in federated learning with training and evaluation methods.

    Subclasses must implement train and evaluate for specific tasks (classification/regression).
    """

    def __init__(
        self, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device
    ) -> None:
        """
        Initializes the model wrapper with PyTorch components.

        Args:
            model (nn.Module): The neural network model.
            optimizer (torch.optim.Optimizer): Optimizer for training.
            criterion (nn.Module): Loss function.
            device (torch.device): Compute device (CPU/GPU).

        Returns:
            None
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    @abstractmethod
    def train(self) -> None:
        """
        Performs training on the model using the optimizer and criterion.

        Must be implemented by subclasses to handle data loaders and epochs.

        Args:
            None

        Returns:
            None

        Raises:
            NotImplementedError: If not overridden.
        """
        pass

    @abstractmethod
    def evaluate(self) -> None:
        """
        Evaluates the model using the criterion.

        Must be implemented by subclasses to compute metrics on data loaders.

        Args:
            None

        Returns:
            None

        Raises:
            NotImplementedError: If not overridden.
        """
        pass
