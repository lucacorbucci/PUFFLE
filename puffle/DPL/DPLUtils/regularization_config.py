from dataclasses import dataclass

import torch


@dataclass
class RegularizationConfig:
    # The number of epochs that we want to use during the training
    epochs: int
    # device on which we want to train the model
    device: torch.device
    batch_size: int
    # seed to use during the training
    seed: int
    optimizer: str
    # regularization is True if we want to use the regularization to
    # reduce the unfairness of the model during the training
    regularization: bool = False
    # lambda is the value of the lambda that we want to use
    # during the training. This value is fixed at the beginning when using starting_lambda_mode = fixed
    regularization_lambda: float = 0
    # fixed or tunable
    regularization_mode: str = None
    # momentum is used to update the Lambda when using the Tunable approach
    momentum: float = None
    # how fast we want to update the Lambda
    # (this is only used when using Tunable approach)
    alpha: float = None
    # Change to a value different to None if we want to
    #  use weight decay when updating alpha
    weight_decay_alpha: float = None
    target: float = None
