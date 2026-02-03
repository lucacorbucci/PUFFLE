"""
Dataclass holding all configuration parameters for federated learning setup.

Includes client numbers, rounds, device/silo settings, sampling fractions, seeds, dataset info, partitioning, training hyperparameters, and preprocessors (scaler/encoder).
"""

from dataclasses import dataclass
from typing import Any

from sklearn.preprocessing import TargetEncoder


@dataclass
class Preferences:
    num_clients: int | None = None
    num_rounds: int | None = None
    cross_device: bool = False
    num_test_nodes: int | None = None
    num_validation_nodes: int | None = None
    num_train_nodes: int | None = None
    num_epochs: int = 1
    sampled_validation_nodes_per_round: float | None = None
    sampled_training_nodes_per_round: float | None = None
    sampled_test_nodes_per_round: float | None = None
    seed: int = 42
    node_shuffle_seed: int | None = None
    fed_dir: str | None = None
    fl_setting: str | None = None
    dataset_path: str | None = None
    sweep: bool = False
    project_name: str | None = None
    run_name: str | None = None
    wandb: bool = False
    dataset_name: str | None = None
    scaler: Any = None
    partitioner_type: str | None = None
    partitioner_alpha: float | None = None
    partitioner_by: str | None = None
    # Fairness Partitioner specific
    sensitive_attribute: str | None = None
    target_attribute: str | None = None
    ratio_unfair_clients: float | None = None
    group_to_reduce: Any = None
    group_to_increment: Any = None
    ratio_unfairness: Any = None
    encoder: TargetEncoder | None = None

    task: str = "classification"

    batch_size: int = 32
    lr: float = 0.01
    optimizer: str = "adam"
    momentum: float = 0.9
    weight_decay: float = 1e-5

    image_path: str | None = None

    # Model architecture parameters
    model: str | None = None
    num_classes: int | None = None
    in_channels: int | None = None

    # Unfairness reduction parameters
    unfairness_reduction: bool = False
    regularization_lambda: float = 0.0
    fairness_metric: str = "disparity"
    regularization_mode: str = "fixed"  # "fixed" or "tunable
    target: float | None = None
    alpha: float | None = None
    weight_decay_alpha: float | None = None

    private_training: bool = False
    epsilon: float | None = None
    epsilon_statistics: float | None = None
    epsilon_lambda: float | None = None
    noise_multiplier: float = 0.0
    max_grad_norm: float = 1000000
