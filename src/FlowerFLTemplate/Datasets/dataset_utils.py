from typing import Any

import numpy as np
import pandas as pd
from flwr.client import Client
from flwr.common import Context
from torch.utils.data import DataLoader

from FlowerFLTemplate.Client.client import FlowerClient
from FlowerFLTemplate.Datasets.abalone import (
    AbaloneDataset,
    prepare_abalone,
    prepare_abalone_for_cross_silo,
)
from FlowerFLTemplate.Datasets.celeba import (
    prepare_celeba,
    prepare_celeba_for_cross_silo,
)
from FlowerFLTemplate.Datasets.dutch import (
    DutchDataset,
    prepare_dutch,
    prepare_dutch_for_cross_silo,
)
from FlowerFLTemplate.Datasets.income import prepare_income_for_cross_silo
from FlowerFLTemplate.Datasets.mnist import (
    prepare_mnist,
    prepare_mnist_for_cross_silo,
)
from FlowerFLTemplate.Utils.preferences import Preferences

# Define TypeAlias for data info return type
DataInfo = dict[str, Any]


def get_data_info(preferences: Preferences) -> DataInfo:
    """
    Get dataset specific information.
    """
    if preferences.dataset_name == "dutch":
        return {"scaler": None, "data_type": "tabular"}  # Scalar is fit on train
    if preferences.dataset_name == "abalone":
        return {"scaler": None, "data_type": "tabular"}
    if preferences.dataset_name == "celeba":
        return {"scaler": None, "data_type": "image"}
    if preferences.dataset_name == "mnist":
        return {"scaler": None, "data_type": "vision"}
    if preferences.dataset_name == "income":
        return {"scaler": None, "encoder": None, "data_type": "tabular"}

    return {}


def get_model_info_from_dataset(dataset_name: str) -> dict[str, int]:
    """
    Returns model architecture information based on dataset name.
    """
    if dataset_name == "dutch":
        return {"in_channels": 12, "num_classes": 2}
    if dataset_name == "abalone":
        return {"in_channels": 8, "num_classes": 1}
    if dataset_name == "income":  # acs_income
        return {"in_channels": 10, "num_classes": 2}
    if dataset_name == "celeba":
        return {
            "in_channels": 3,
            "num_classes": 2,
        }  # Attributes to predict is usually 1 (e.g. smiling) but CelebaNet might expect 2 output for CrossEntropy? Or is it Binary? Checking CelebaNet usage.
    if dataset_name == "mnist":
        return {"in_channels": 1, "num_classes": 10, "pixel": 28}
    return {}


def prepare_data_for_cross_device(
    context: Context,
    partition: pd.DataFrame | None,
    preferences: Preferences,
    partition_id: int,
) -> Client:
    """
    Prepare data and client for cross-device setting.
    """
    if preferences.dataset_name == "dutch":
        if partition is None:
            msg = "Partition cannot be None for Dutch dataset in cross-device"
            raise ValueError(msg)

        if not isinstance(partition, pd.DataFrame):
            partition = pd.DataFrame(partition)

        # Split train/val
        partition = partition.sample(frac=1, random_state=preferences.seed).reset_index(
            drop=True
        )
        split_idx = int(0.8 * len(partition))
        train_df = partition.iloc[:split_idx]
        val_df = partition.iloc[split_idx:]

        # Prepare train
        # Note: Scaler handling in FL is tricky. Ideally strict FL uses no global scaler.
        # Here we fit scaler on local train data.
        x_train, z_train, y_train, scaler = prepare_dutch(train_df, scaler=None)

        # Prepare val (use scaler from train)
        x_val, z_val, y_val, _ = prepare_dutch(val_df, scaler=scaler)

        train_ds = DutchDataset(
            x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(np.float32),
            z=z_train.astype(np.float32),
            y=y_train.astype(np.float32),
        )
        val_ds = DutchDataset(
            x=np.hstack((x_val, np.ones((x_val.shape[0], 1)))).astype(np.float32),
            z=z_val.astype(np.float32),
            y=y_val.astype(np.float32),
        )

        trainloader = DataLoader(
            train_ds, batch_size=preferences.batch_size, shuffle=True
        )
        valloader = DataLoader(val_ds, batch_size=preferences.batch_size, shuffle=False)

        return FlowerClient(
            trainloader=trainloader,
            valloader=valloader,
            preferences=preferences,
            partition_id=partition_id,
        ).to_client()

    if preferences.dataset_name == "abalone":
        if partition is None:
            msg = "Partition cannot be None for Abalone dataset"
            raise ValueError(msg)

        # Similar logic for Abalone
        partition = partition.sample(frac=1, random_state=preferences.seed).reset_index(
            drop=True
        )
        split_idx = int(0.8 * len(partition))
        train_df = partition.iloc[:split_idx]
        val_df = partition.iloc[split_idx:]

        x_train, y_train, scaler = prepare_abalone(train_df, scaler=None)
        x_val, y_val, _ = prepare_abalone(val_df, scaler=scaler)

        train_ds = AbaloneDataset(x=x_train, y=y_train)
        val_ds = AbaloneDataset(x=x_val, y=y_val)

        trainloader = DataLoader(
            train_ds, batch_size=preferences.batch_size, shuffle=True
        )
        valloader = DataLoader(val_ds, batch_size=preferences.batch_size, shuffle=False)

        return FlowerClient(
            trainloader=trainloader,
            valloader=valloader,
            preferences=preferences,
            partition_id=partition_id,
        ).to_client()

    if preferences.dataset_name == "mnist":
        # Prepare MNIST
        trainloader = prepare_mnist(partition, preferences)
        # For now reuse trainloader as valloader or split if supported
        return FlowerClient(
            trainloader=trainloader,
            valloader=trainloader,
            preferences=preferences,
            partition_id=partition_id,
        ).to_client()

    if preferences.dataset_name == "celeba":
        if partition is None:
            msg = "Partition cannot be None for Celeba dataset"
            raise ValueError(msg)
        trainloader = prepare_celeba(partition, preferences)
        return FlowerClient(
            trainloader=trainloader,
            valloader=trainloader,
            preferences=preferences,
            partition_id=partition_id,
        ).to_client()

    msg = f"Dataset {preferences.dataset_name} not supported for cross-device yet"
    raise ValueError(msg)


def prepare_data_for_cross_silo(
    context: Context,
    partition: pd.DataFrame | None,
    preferences: Preferences,
    partition_id: int,
) -> Client:
    """
    Prepares data for cross-silo federated learning by delegating to dataset-specific functions.
    """
    if preferences.dataset_name == "dutch":
        return prepare_dutch_for_cross_silo(preferences, partition, partition_id)
    if preferences.dataset_name == "mnist":
        return prepare_mnist_for_cross_silo(preferences, partition, partition_id)
    if preferences.dataset_name == "abalone":
        if partition is None:
            msg = "Partition cannot be None for Abalone dataset"
            raise ValueError(msg)
        return prepare_abalone_for_cross_silo(preferences, partition, partition_id)
    if preferences.dataset_name == "income":
        return prepare_income_for_cross_silo(preferences, partition_id)
    if preferences.dataset_name == "celeba":
        if partition is None:
            msg = "Partition cannot be None for Celeba dataset"
            raise ValueError(msg)
        return prepare_celeba_for_cross_silo(preferences, partition, partition_id)

    msg = f"Unsupported dataset: {preferences.dataset_name}"
    raise ValueError(msg)


def partition_data(
    data: pd.DataFrame,
    num_clients: int,
    partitioner_type: str,
    partitioner_alpha: float = 1.0,
    seed: int = 42,
) -> dict[int, pd.DataFrame]:
    """Partition the dataframe indices among clients."""
    rng = np.random.default_rng(seed)
    n_samples = len(data)
    indices = np.arange(n_samples)

    if partitioner_type in {"iid", "dirichlet"}:
        rng.shuffle(indices)
        partitions = np.array_split(indices, num_clients)
    else:
        rng.shuffle(indices)
        partitions = np.array_split(indices, num_clients)

    return {i: data.iloc[p] for i, p in enumerate(partitions)}


def load_partitioned_dataset(
    dataset_name: str,
    dataset_path: str,
    num_clients: int,
    batch_size: int,
    partitioner_type: str = "iid",
    partitioner_alpha: float = 1.0,
    partitioner_by: str | None = None,
    seed: int = 42,
    fed_dir: str = "",
) -> dict[int, dict[str, Any]]:
    """
    Legacy method for explicit loading.
    """
    loaders = {}
    return loaders
