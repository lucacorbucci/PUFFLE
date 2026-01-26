from typing import Any

import numpy as np
import pandas as pd
from Client.client import FlowerClient
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from Utils.preferences import Preferences


class DutchDataset(Dataset):
    def __init__(self, x: np.ndarray, z: np.ndarray, y: np.ndarray) -> None:
        """
        Initializes the DutchDataset with features, sensitive attributes, and targets.

        Stores samples, sensitive features, targets, and indexes for dataset access.

        Args:
            x (np.ndarray): Feature data array.
            z (np.ndarray): Sensitive attribute data array.
            y (np.ndarray): Target data array.

        Returns:
            None
        """
        self.samples = x
        self.sensitive_features = z
        self.targets = y
        self.indexes = range(len(self.samples))

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Args:
            None

        Returns:
            int: Size of the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any, int, int]:
        """
        Retrieves a single sample from the dataset by index.

        Returns feature, sensitive attribute, and target for the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[Any, Any, Any]: (feature sample, sensitive sample, target sample)
        """
        x_sample = self.samples[idx]
        z_sample = self.sensitive_features[idx]
        y_sample = self.targets[idx]

        return x_sample, z_sample, y_sample, self.indexes[idx], idx


def get_dutch_scaler(
    sweep: bool,
    seed: int,
    dutch_df: pd.DataFrame | None = None,
    validation_seed: int | None = None,
) -> MinMaxScaler:
    """
    Computes and returns a MinMaxScaler fitted on the Dutch dataset.

    Validates input DataFrame and uses prepare_dutch to obtain the scaler.

    Args:
        sweep (bool): Whether in hyperparameter sweep mode.
        seed (int): Random seed for reproducibility.
        dutch_df (pd.DataFrame | None): Dutch data DataFrame; required.
        validation_seed (int | None): Seed for validation split.

    Returns:
        MinMaxScaler: Fitted scaler instance.

    Raises:
        ValueError: If dutch_df is None.
    """
    if dutch_df is None:
        raise ValueError("dutch_df cannot be None")

    _, _, _, scaler = prepare_dutch(
        dutch_df=dutch_df,
    )
    return scaler


def prepare_dutch(
    dutch_df: pd.DataFrame,
    scaler: MinMaxScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Preprocesses Dutch DataFrame: handles missing values, binarizes sex/occupation, one-hot encodes categoricals, scales features.

    Fills missing values with mode, creates binary targets/sensitive, applies MinMaxScaler if not provided.

    Args:
        dutch_df (pd.DataFrame): Input Dutch data.
        scaler (MinMaxScaler | None): Pre-fitted scaler; if None, fits a new one.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]: Scaled features, sensitive array, target array, scaler.

    Raises:
        ValueError: If missing values persist after filling.
    """
    # check the columns with missign values:
    missing_values_columns = dutch_df.columns[dutch_df.isna().any()].tolist()
    for column in missing_values_columns:
        dutch_df[column] = dutch_df[column].fillna(dutch_df[column].mode()[0])

    if len(dutch_df.columns[dutch_df.isna().any()].tolist()) != 0:
        error_message = "There are still missing values in the dataset"
        raise ValueError(error_message)

    dutch_df["sex_binary"] = np.where(dutch_df["sex"] == 1, 1, 0)
    dutch_df["occupation_binary"] = np.where(dutch_df["occupation"] >= 300, 1, 0)

    del dutch_df["sex"]
    del dutch_df["occupation"]

    y_train = dutch_df["occupation_binary"].astype(int).values
    z_train = dutch_df["sex_binary"].astype(int).values
    del dutch_df["occupation_binary"]
    dutch_df = pd.get_dummies(dutch_df, columns=None, drop_first=False)

    if scaler is None:
        scaler = MinMaxScaler()
    x_train = scaler.fit_transform(dutch_df)

    return x_train, np.array(z_train), np.array(y_train), scaler


def prepare_dutch_for_cross_silo(preferences: Preferences, partition: Any, partition_id: int) -> Any:
    """
    Prepares Dutch data for cross-silo federated learning from a partition.

    Splits into train/test (20% test), optionally train/val for sweep; processes with prepare_dutch, creates DataLoaders and FlowerClient. Prints debug info for partition_id 0/1.

    Args:
        preferences (Preferences): FL configuration including batch_size, seed, scaler.
        partition (Any): Data partition for this client.
        partition_id (int): Client ID for debug printing.

    Returns:
        Any: FlowerClient instance wrapped as .to_client().

    Raises:
        ValueError: If data processing fails.
    """
    partition_train_test = partition.train_test_split(test_size=0.2, seed=preferences.seed)

    test = partition_train_test["test"]
    if partition_id == 0:
        print(f"partition Id {partition_id} - Test set size: {len(test)}")
        print(f"Test set columns: {test.column_names}")
        print(f"Test set example:\n{test[0]}")
    elif partition_id == 1:
        print(f"partition Id {partition_id} - Test set size: {len(test)}")
        print(f"Test set columns: {test.column_names}")
        print(f"Test set example:\n{test[0]}")

    if preferences.sweep:
        print("[Preparing data for cross-silo for sweep...]")

        partition_loader_train_val = partition_train_test["train"].train_test_split(
            test_size=0.2, seed=preferences.node_shuffle_seed
        )
        train = partition_loader_train_val["train"].to_pandas()
        val = partition_loader_train_val["test"].to_pandas()

        x_train, z_train, y_train, _ = prepare_dutch(
            dutch_df=train,
            scaler=preferences.scaler,
        )

        x_val, z_val, y_val, _ = prepare_dutch(
            dutch_df=val,
            scaler=preferences.scaler,
        )

        train_dataset = DutchDataset(
            x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(np.float32),
            z=z_train.astype(np.float32),
            y=y_train.astype(np.float32),
        )
        val_dataset = DutchDataset(
            x=np.hstack((x_val, np.ones((x_val.shape[0], 1)))).astype(np.float32),
            z=z_val.astype(np.float32),
            y=y_val.astype(np.float32),
        )

        trainloader = DataLoader(train_dataset, batch_size=preferences.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=preferences.batch_size, shuffle=False)

        return FlowerClient(
            trainloader=trainloader, valloader=val_loader, preferences=preferences, partition_id=partition_id
        ).to_client()
    print("[Preparing data for cross-silo...]")
    train = partition_train_test["train"].to_pandas()
    test = partition_train_test["test"].to_pandas()

    x_train, z_train, y_train, _ = prepare_dutch(
        dutch_df=train,
        scaler=preferences.scaler,
    )

    x_test, z_test, y_test, _ = prepare_dutch(
        dutch_df=test,
        scaler=preferences.scaler,
    )

    train_dataset = DutchDataset(
        x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(np.float32),
        z=z_train.astype(np.float32),
        y=y_train.astype(np.float32),
    )
    test_dataset = DutchDataset(
        x=np.hstack((x_test, np.ones((x_test.shape[0], 1)))).astype(np.float32),
        z=z_test.astype(np.float32),
        y=y_test.astype(np.float32),
    )

    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    trainloader = DataLoader(train_dataset, batch_size=preferences.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=preferences.batch_size, shuffle=False)

    return FlowerClient(
        trainloader=trainloader, valloader=test_loader, preferences=preferences, partition_id=partition_id
    ).to_client()
