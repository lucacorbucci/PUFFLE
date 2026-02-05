import os
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from flwr.common import Context
from torch.utils.data import DataLoader

from FlowerFLTemplate.Client.client import FlowerClient
from FlowerFLTemplate.Datasets.abalone import (
    AbaloneDataset,
    get_abalone_scaler,
    prepare_abalone,
    prepare_abalone_for_cross_silo,
)
from FlowerFLTemplate.Datasets.celeba import (
    prepare_celeba,
    prepare_celeba_for_cross_silo,
)
from FlowerFLTemplate.Datasets.dutch import (
    DutchDataset,
    get_dutch_scaler,
    prepare_dutch_fl,
    prepare_dutch_for_cross_silo,
)
from FlowerFLTemplate.Datasets.income import (
    get_income_scaler,
    prepare_income_for_cross_silo,
)
from FlowerFLTemplate.Datasets.mnist import (
    download_mnist,
    prepare_mnist,
    prepare_mnist_for_cross_silo,
)
from FlowerFLTemplate.Utils.preferences import Preferences


def get_data_info(preferences: Preferences) -> dict[str, Any]:
    """
    Retrieves dataset-specific information including data type, target, sensitive attributes, and preprocessors (scaler/encoder).

    Supports "dutch", "mnist", "abalone", "income" datasets; loads data if needed and computes scalers/encoders.

    Args:
        preferences (Preferences): Configuration containing dataset_name, dataset_path, sweep, seed, etc.

    Returns:
        dict[str, Any]: Dictionary with keys like "data_type", "target", "sensitive_attribute", "scaler", "encoder".

    Raises:
        ValueError: If unsupported dataset_name.

    """
    match preferences.dataset_name:
        case "dutch":
            df = pd.read_csv(preferences.dataset_path)
            scaler = get_dutch_scaler(
                sweep=preferences.sweep,
                seed=preferences.seed,
                dutch_df=df,
                validation_seed=preferences.node_shuffle_seed,
            )

            return {
                "data_type": "csv",
                "target": "occupation",
                "sensitive_attribute": "sex",
                "scaler": scaler,
            }

        case "mnist":
            if not os.path.exists(os.path.join(preferences.dataset_path)):
                download_mnist()
            return {"data_type": "imagefolder"}
        case "abalone":
            df = pd.read_csv(preferences.dataset_path)
            scaler = get_abalone_scaler(
                sweep=preferences.sweep,
                seed=preferences.seed,
                abalone_df=df,
                validation_seed=preferences.node_shuffle_seed,
            )

            return {"data_type": "csv", "target": "Rings", "scaler": scaler}
        case "income":
            # open all the csv files in the directory and concatenate them into a single dataframe
            all_files = []
            for file_name in os.listdir(preferences.dataset_path):
                # check if the file is a folder
                if os.path.isdir(os.path.join(preferences.dataset_path, file_name)):
                    all_files.extend(
                        os.path.join(preferences.dataset_path, file_name, f)
                        for f in os.listdir(
                            os.path.join(preferences.dataset_path, file_name)
                        )
                        if f.endswith(".csv")
                    )

            df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
            scaler, encoder = get_income_scaler(
                sweep=preferences.sweep,
                seed=preferences.seed,
                df=df,
                validation_seed=preferences.node_shuffle_seed,
            )
            return {
                "data_type": "csv",
                "target": ">50K",
                "sensitive_attribute": "sex",
                "scaler": scaler,
                "encoder": encoder,
            }
        case "celeba":
            return {
                "data_type": "csv",
                "target": "Smiling",
                "sensitive_attribute": "Male",
            }
        case _:
            msg = f"Unsupported dataset: {preferences.dataset_name}"
            raise ValueError(msg)


def get_model_info_from_dataset(dataset_name: str) -> dict[str, int]:
    """
    Returns model architecture information based on dataset name.
    """
    if dataset_name == "dutch":
        return {"in_channels": 11, "num_classes": 2}
    if dataset_name == "abalone":
        return {"in_channels": 8, "num_classes": 1}
    if dataset_name == "income":  # acs_income
        return {"in_channels": 10, "num_classes": 2}
    if dataset_name == "celeba":
        return {
            "in_channels": 3,
            "num_classes": 2,
        }
    if dataset_name == "mnist":
        return {"in_channels": 1, "num_classes": 10, "pixel": 28}
    return {}


def _create_dutch_dataloaders(
    partition: Any, preferences: Preferences
) -> tuple[DataLoader, DataLoader]:
    """Create DataLoaders for Dutch dataset from a partition."""
    train = partition.to_pandas()
    x_train, z_train, y_train, _ = prepare_dutch_fl(
        dutch_df=train,
        scaler=preferences.scaler,
    )
    train_dataset = DutchDataset(
        x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(np.float32),
        z=z_train.astype(np.float32),
        y=y_train.astype(np.float32),
    )

    trainloader = DataLoader(
        train_dataset, batch_size=preferences.batch_size, shuffle=True
    )
    # For cross-device, use same loader for train/val
    return trainloader, trainloader


def _create_abalone_dataloaders(
    partition: Any, preferences: Preferences
) -> tuple[DataLoader, DataLoader]:
    """Create DataLoaders for Abalone dataset from a partition."""
    train = partition.to_pandas()
    x_train, y_train, _ = prepare_abalone(
        abalone_df=train,
        scaler=preferences.scaler,
    )
    train_dataset = AbaloneDataset(
        x=x_train,
        y=y_train,
    )
    trainloader = DataLoader(
        train_dataset, batch_size=preferences.batch_size, shuffle=True
    )
    return trainloader, trainloader


def _create_celeba_dataloaders(
    partition: Any, preferences: Preferences
) -> tuple[DataLoader, DataLoader]:
    """Create DataLoaders for CelebA dataset from a partition."""
    train = partition.to_pandas()
    trainloader = prepare_celeba(train, preferences)
    return trainloader, trainloader


def prepare_data_for_cross_device(
    context: Context,
    partition: Any,
    preferences: Preferences,
    partition_id: int,
    partitioner: Any = None,
) -> Any:
    """
    Prepares data for cross-device federated learning from a partition.

    Uses lazy loading: the actual data loading is deferred until first fit()/evaluate().

    Args:
        context (Context): Flower context (unused).
        partition (Any): Data partition for this client (may be None if using lazy loading).
        preferences (Preferences): FL configuration including dataset_name, batch_size, scaler.
        partition_id (int): Client partition ID.
        partitioner (Any): Optional partitioner for lazy loading.

    Returns:
        Any: FlowerClient instance wrapped as .to_client().

    Raises:
        ValueError: If unsupported dataset_name.

    """

    # Create a lazy loader function that captures the partition or partitioner
    def create_data_loader_fn() -> Callable[[], tuple[DataLoader, DataLoader]]:
        def load_data() -> tuple[DataLoader, DataLoader]:
            # Load partition if we have a partitioner
            data_partition = partition
            if data_partition is None and partitioner is not None:
                data_partition = partitioner.load_partition(partition_id)

            if preferences.dataset_name == "dutch":
                return _create_dutch_dataloaders(data_partition, preferences)
            if preferences.dataset_name == "mnist":
                trainloader = prepare_mnist(data_partition, preferences)
                return trainloader, trainloader
            if preferences.dataset_name == "abalone":
                return _create_abalone_dataloaders(data_partition, preferences)
            if preferences.dataset_name == "celeba":
                return _create_celeba_dataloaders(data_partition, preferences)
            msg = f"Unsupported dataset: {preferences.dataset_name}"
            raise ValueError(msg)

        return load_data

    return FlowerClient(
        partition_id=partition_id,
        preferences=preferences,
        data_loader_fn=create_data_loader_fn(),
    ).to_client()


def prepare_data_for_cross_silo(
    context: Context,
    partition: Any,
    preferences: Preferences,
    partition_id: int,
    partitioner: Any = None,
) -> Any:
    """
    Prepares data for cross-silo federated learning by delegating to dataset-specific functions.

    Supports "dutch", "mnist", "abalone", "income"; for income, partition unused.

    Args:
        context (Context): Flower context (unused).
        partition (Any): Data partition for this client (unused for income).
        preferences (Preferences): FL configuration including dataset_name.
        partition_id (int): Client partition ID (passed to specific functions).
        partitioner (Any): Optional partitioner for lazy loading (not used in cross-silo).

    Returns:
        Any: FlowerClient instance from specific preparation function.

    Raises:
        ValueError: If unsupported dataset_name.

    """
    # Cross-silo functions handle their own data loading
    # These are kept as-is for now since they have more complex train/val splits
    if preferences.dataset_name == "dutch":
        return prepare_dutch_for_cross_silo(preferences, partition, partition_id)
    if preferences.dataset_name == "mnist":
        return prepare_mnist_for_cross_silo(preferences, partition, partition_id)
    if preferences.dataset_name == "abalone":
        return prepare_abalone_for_cross_silo(preferences, partition, partition_id)
    if preferences.dataset_name == "income":
        return prepare_income_for_cross_silo(preferences, partition_id)
    if preferences.dataset_name == "celeba":
        return prepare_celeba_for_cross_silo(preferences, partition, partition_id)

    msg = f"Unsupported dataset: {preferences.dataset_name}"
    raise ValueError(msg)
