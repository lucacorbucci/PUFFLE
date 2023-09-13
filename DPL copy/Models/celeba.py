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


# class CelebaNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 64, 3, 1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(64, 128, 3, 1)
#         self.conv3 = nn.Conv2d(128, 256, 3, 1)
#         self.conv4 = nn.Conv2d(256, 512, 3, 1)
#         self.fc1 = nn.Linear(14 * 14 * 512, 1024)
#         self.fc2 = nn.Linear(1024, 256)
#         self.fc3 = nn.Linear(256, 2)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool(x)
#         x = self.relu(x)

#         x = self.conv2(x)
#         x = self.pool(x)
#         x = self.relu(x)

#         x = self.conv3(x)
#         x = self.pool(x)
#         x = self.relu(x)

#         x = self.conv4(x)
#         x = self.pool(x)
#         x = self.relu(x)
#         x = x.view(-1, 14 * 14 * 512)
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.fc3(x)
#         return x
