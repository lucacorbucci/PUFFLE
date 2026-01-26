import torch.nn.functional as functional
from torch import Tensor, nn


class LinearClassificationNet(nn.Module):
    """
    A fully-connected single-layer linear NN for classification.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Initializes a single-layer linear neural network for classification.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output classes.

        Returns:
            None
        """
        super().__init__()
        self.layer1 = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs forward pass through the linear layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            Tensor: Output logits of shape (batch_size, output_size).
        """
        x = self.layer1(x.float())
        return x


class AbaloneNet(nn.Module):
    """Neural Network for Abalone age prediction"""

    def __init__(self, input_size: int, hidden_sizes: list[int] | None = None, dropout_rate: float = 0.2) -> None:
        """
        Initializes a multi-layer feedforward network for Abalone regression.

        Builds sequential layers with Linear, ReLU, BatchNorm1d, Dropout; output layer is linear without activation.

        Args:
            input_size (int): Number of input features.
            hidden_sizes (list[int] | None): List of hidden layer sizes. Defaults to [128, 64, 32].
            dropout_rate (float): Dropout probability. Defaults to 0.2.

        Returns:
            None
        """
        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]
        super().__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.BatchNorm1d(hidden_size), nn.Dropout(dropout_rate)]
            )
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs forward pass through the network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            Tensor: Output prediction of shape (batch_size, 1).
        """
        return self.network(x)


class SimpleMNISTModel(nn.Module):
    """
    A simpler fully connected model for MNIST.
    """

    def __init__(self, num_classes: int = 10) -> None:
        """
        Initializes a simple two-layer fully connected network for MNIST classification.

        Input layer: 784 -> 128 with ReLU; output: 128 -> num_classes.

        Args:
            num_classes (int, optional): Number of output classes. Defaults to 10.

        Returns:
            None
        """
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs forward pass: flattens input, applies ReLU after first layer, linear output.

        Args:
            x (Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
            Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Flatten the input
        x = x.view(-1, 28 * 28)

        # Pass through layers
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)

        return x


class CelebaNet(nn.Module):
    """This class defines the CelebaNet."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        dropout_rate: float = 0,
    ) -> None:
        """
        Initializes the CelebaNet CNN for CelebA classification.

        Three conv layers with ReLU and MaxPool; fully connected output. Dropout not used.

        Args:
            in_channels (int, optional): Input channels. Defaults to 3 (RGB).
            num_classes (int, optional): Output classes. Defaults to 2.
            dropout_rate (float, optional): Dropout rate (unused). Defaults to 0.

        Returns:
            None
        """
        super().__init__()
        self.cnn1 = nn.Conv2d(
            in_channels,
            8,
            kernel_size=(3, 3),
            padding=(1, 1),
            stride=(1, 1),
        )
        self.cnn2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.cnn3 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        self.fc1 = nn.Linear(2048, 2)
        self.gn_relu = nn.Sequential(
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        # self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_data: Tensor) -> Tensor:
        """
        Performs forward pass: three conv + ReLU + MaxPool, flatten, linear output.

        Args:
            input_data (Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Tensor: Output logits of shape (batch_size, num_classes).
        """
        out = self.gn_relu(self.cnn1(input_data))
        out = self.gn_relu(self.cnn2(out))
        out = self.gn_relu(self.cnn3(out))
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out
