from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TrainParameters:
    epochs: int
    device: torch.device
    criterion: torch.nn.Module
    wandb_run: Any
    batch_size: int
    seed: int
    epsilon: float
    DPL_lambda: float = 0
    private: bool = False
    DPL: bool = False
    noise_multiplier: float = None
    probability_estimation: bool = False
    perfect_probability_estimation: bool = False
    partition_ratio:int = None
