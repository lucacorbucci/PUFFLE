import os
from typing import Any

import numpy as np
import pandas as pd
from Client.client import FlowerClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, TargetEncoder
from torch.utils.data import DataLoader, Dataset
from Utils.preferences import Preferences


class IncomeDataset(Dataset):
    def __init__(self, x: np.ndarray, z: np.ndarray, y: np.ndarray) -> None:
        """
        Initializes the IncomeDataset with features, sensitive attributes, and targets.

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

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any]:
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

        return x_sample, z_sample, y_sample


def get_income_scaler(
    sweep: bool,
    seed: int,
    df: pd.DataFrame | None = None,
    validation_seed: int | None = None,
) -> tuple[StandardScaler, TargetEncoder]:
    """
    Computes and returns StandardScaler and TargetEncoder fitted on the Income dataset.

    Validates input DataFrame and uses prepare_income to obtain preprocessors.

    Args:
        sweep (bool): Whether in hyperparameter sweep mode.
        seed (int): Random seed for reproducibility.
        df (pd.DataFrame | None): Income data DataFrame; required.
        validation_seed (int | None): Seed for validation split.

    Returns:
        tuple[StandardScaler, TargetEncoder]: Fitted scaler and encoder instances.

    Raises:
        ValueError: If df is None or not a DataFrame.
    """
    if df is None:
        error = "df cannot be None"
        raise ValueError(error)
    if type(df) is not pd.DataFrame:
        error = "df must be a pandas DataFrame"
        raise ValueError(error)

    _, _, _, scaler, encoder = prepare_income(
        df=df,
    )
    return scaler, encoder


def prepare_income(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
    encoder: TargetEncoder | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, TargetEncoder]:
    """
    Preprocesses Income DataFrame: encodes categoricals with TargetEncoder, scales continuous features with StandardScaler.

    Drops target/sensitive columns, applies encoders/scalers if not provided.

    Args:
        df (pd.DataFrame): Input Income data.
        scaler (StandardScaler | None): Pre-fitted scaler; if None, fits a new one.
        encoder (TargetEncoder | None): Pre-fitted encoder; if None, fits a new one.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, TargetEncoder]: Features array, sensitive array, target array, scaler, encoder.

    Raises:
        ValueError: If DataFrame is invalid.
    """
    # Separate features and target

    categorical_columns = ["COW", "SCHL", "RAC1P"]
    continuous_columns = ["AGEP", "WKHP", "OCCP", "POBP", "RELP"]

    # get the target and sensitive attributes
    target_attributes = df[">50K"].values
    sensitive_attributes = df["SEX"].values

    df = df.drop(
        columns=[
            ">50K",
            "SEX",
        ]
    )

    if encoder is None:
        encoder = TargetEncoder(smooth="auto").fit(df[categorical_columns], target_attributes)
    df[categorical_columns] = encoder.transform(df[categorical_columns])

    # normalize the continuous using standard scaler
    if scaler is None:
        scaler = StandardScaler().fit(df[continuous_columns])
    df[continuous_columns] = scaler.transform(df[continuous_columns])

    # convert to numpy arrays
    x_train = df.to_numpy(dtype=np.float32)

    return x_train, np.array(sensitive_attributes), np.array(target_attributes), scaler, encoder


def prepare_income_for_cross_silo(preferences: Preferences, partition_id: int) -> Any:
    """
    Prepares Income data for cross-silo federated learning for a specific partition.

    Loads train/test CSV files from partition directory, optionally splits train into train/val for sweep; processes with prepare_income, creates DataLoaders and FlowerClient.

    Args:
        preferences (Preferences): FL configuration including dataset_path, batch_size, seed, scaler, encoder.
        partition_id (int): Client partition ID for loading specific files.

    Returns:
        Any: FlowerClient instance wrapped as .to_client().

    Raises:
        FileNotFoundError: If CSV files not found in partition directory.
        ValueError: If data processing fails.
    """
    path = f"{preferences.dataset_path}/{partition_id}/"
    for file in os.listdir(path):
        if file.endswith(".csv"):
            if "train" in file:
                train = pd.read_csv(f"{preferences.dataset_path}/{partition_id}/{file}")
            elif "test" in file:
                test = pd.read_csv(f"{preferences.dataset_path}/{partition_id}/{file}")

    if preferences.sweep:
        print("[Preparing data for cross-silo for sweep...]")

        train, val = train_test_split(train, test_size=0.2, random_state=preferences.node_shuffle_seed)

        x_train, z_train, y_train, _, _ = prepare_income(
            df=train,
            scaler=preferences.scaler,
            encoder=preferences.encoder,
        )

        x_val, z_val, y_val, _, _ = prepare_income(
            df=val,
            scaler=preferences.scaler,
            encoder=preferences.encoder,
        )

        train_dataset = IncomeDataset(
            x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(np.float32),
            z=z_train.astype(np.float32),
            y=y_train.astype(np.float32),
        )
        val_dataset = IncomeDataset(
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

    x_train, z_train, y_train, _, _ = prepare_income(
        df=train,
        scaler=preferences.scaler,
        encoder=preferences.encoder,
    )

    x_test, z_test, y_test, _, _ = prepare_income(
        df=test,
        scaler=preferences.scaler,
        encoder=preferences.encoder,
    )

    train_dataset = IncomeDataset(
        x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(np.float32),
        z=z_train.astype(np.float32),
        y=y_train.astype(np.float32),
    )
    test_dataset = IncomeDataset(
        x=np.hstack((x_test, np.ones((x_test.shape[0], 1)))).astype(np.float32),
        z=z_test.astype(np.float32),
        y=y_test.astype(np.float32),
    )

    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    trainloader = DataLoader(train_dataset, batch_size=preferences.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=preferences.batch_size, shuffle=False)

    return FlowerClient(trainloader=trainloader, valloader=test_loader, preferences=preferences).to_client()
