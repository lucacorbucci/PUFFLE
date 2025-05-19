import torch.nn as nn
from torch import nn
import torch.nn.functional as F


class LinearClassificationNetValerio(nn.Module):
    """
    A fully-connected single-layer linear NN for classification.
    """

    def __init__(self, input_size, output_size):
        super(LinearClassificationNetValerio, self).__init__()
        self.layer1 = nn.Linear(input_size, 64, bias=False)
        self.layer2 = nn.Linear(64, output_size, bias=False)

    def forward(self, x):
        x = F.relu(self.layer1(x.float()))
        x = self.layer2(x)
        return x
