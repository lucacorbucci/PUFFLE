import torch.nn as nn
from torch import Tensor, nn


class CelebaNet(nn.Module):
    """This class defines the CelebaNet."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        dropout_rate: float = 0,
    ) -> None:
        """Initializes the CelebaNet network.

        Args:
        ----
            in_channels (int, optional): Number of input channels . Defaults to 3.
            num_classes (int, optional): Number of classes . Defaults to 2.
            dropout_rate (float, optional): _description_. Defaults to 0.2.
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
        """Defines the forward pass of the network.

        Args:
            input_data (Tensor): Input data

        Returns
        -------
            Tensor: Output data
        """
        out = self.gn_relu(self.cnn1(input_data))
        out = self.gn_relu(self.cnn2(out))
        out = self.gn_relu(self.cnn3(out))
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        return out
