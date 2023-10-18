from dataclasses import dataclass
from typing import Any

import torch
from fl_puf.Utils.enums import StartingLambdaMode


@dataclass
class TrainParameters:
    epochs: int
    device: torch.device
    criterion: torch.nn.Module
    wandb_run: Any
    batch_size: int
    seed: int
    epsilon: float
    starting_lambda_mode: StartingLambdaMode
    momentum: float
    DPL_lambda: float = 0
    private: bool = False
    DPL: bool = False
    noise_multiplier: float = None
    target: float = 0.2
    alpha: float = 0.1
    probability_estimation: bool = False
    perfect_probability_estimation: bool = False
    partition_ratio: int = None
    cross_silo: bool = False
    weight_decay_lambda: float = 0.01
    sweep: bool = False
    optimizer: str = "sgd"
    starting_lambda_value: float = None
