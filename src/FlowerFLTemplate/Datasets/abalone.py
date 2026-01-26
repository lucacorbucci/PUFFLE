from typing import Any

import numpy as np
import pandas as pd
import torch
from Client.client import FlowerClient
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset
from Utils.preferences import Preferences


class AbaloneDataset(Dataset):
    """Custom Dataset for Abalone data"""

    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Initializes the AbaloneDataset with feature and target tensors.

        Converts numpy arrays to PyTorch FloatTensors; reshapes y to (n, 1).

        Args:
            x (np.ndarray): Feature data array.
            y (np.ndarray): Target data array.

        Returns:
            None
        """
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y).view(-1, 1)

    def __len__(self) -> int:
        """
        Returns the size of the dataset.

        Args:
            None

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[Any, Any, Any]:
        """
        Retrieves a sample from the dataset by index.

        Returns feature tensor, dummy label (-1), and target tensor.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[Any, Any, Any]: (feature tensor, -1, target tensor)
        """
        return self.X[idx], -1, self.y[idx]


def get_abalone_scaler(
    sweep: bool,
    seed: int,
    abalone_df: pd.DataFrame | None = None,
    validation_seed: int | None = None,
) -> StandardScaler:
    """
    Fits and returns a StandardScaler for the Abalone dataset.

    Validates input DataFrame and uses prepare_abalone to compute the scaler.

    Args:
        sweep (bool): Whether in hyperparameter sweep mode.
        seed (int): Random seed for reproducibility.
        abalone_df (pd.DataFrame | None): Abalone data DataFrame; required.
        validation_seed (int | None): Seed for validation split.

    Returns:
        StandardScaler: Fitted scaler instance.

    Raises:
        ValueError: If abalone_df is None or not a DataFrame.
    """
    if abalone_df is None:
        error = "abalone_df cannot be None"
        raise ValueError(error)
    if type(abalone_df) is not pd.DataFrame:
        error = "abalone_df must be a pandas DataFrame"
        raise ValueError(error)

    _, _, scaler = prepare_abalone(
        abalone_df=abalone_df,
    )
    return scaler


def prepare_abalone(
    abalone_df: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Preprocesses Abalone DataFrame for training: encodes categorical features, scales numerical ones.

    Separates features/target, encodes 'Sex', fits/transforms scaler if not provided, prints dataset info.

    Args:
        abalone_df (pd.DataFrame): Input Abalone data.
        scaler (StandardScaler | None): Pre-fitted scaler; if None, fits a new one.

    Returns:
        tuple[np.ndarray, np.ndarray, StandardScaler]: Scaled features, target array, scaler instance.

    Raises:
        ValueError: If DataFrame is invalid.
    """
    # Separate features and target
    x = abalone_df.drop("Rings", axis=1)
    y_train = abalone_df["Rings"].values  # Age = Rings + 1.5, but we'll predict rings directly

    # Encode categorical variable (Sex)
    le = LabelEncoder()
    x["Sex"] = le.fit_transform(x["Sex"])

    # Convert to numpy array
    x = x.values

    # Scale the features
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x)

    print(f"\nTraining set size: {x_train.shape[0]}")
    print(f"Number of features: {x_train.shape[1]}")

    return x_train, np.array(y_train), scaler


def prepare_abalone_for_cross_silo(preferences: Preferences, partition: Any, partition_id: int) -> Any:
    """
    Prepares Abalone data for cross-silo federated learning from a partition.

    Splits into train/test (20% test), optionally train/val for sweep; processes with prepare_abalone, creates DataLoaders and FlowerClient.

    Args:
        preferences (Preferences): FL configuration including batch_size, seed, etc.
        partition (Any): Data partition for this client.
        partition_id (int): Client partition ID (unused in implementation).

    Returns:
        Any: FlowerClient instance wrapped as .to_client().

    Raises:
        ValueError: If data processing fails.
    """
    partition_train_test = partition.train_test_split(test_size=0.2, seed=preferences.seed)
    if preferences.sweep:
        print("[Preparing data for cross-silo for sweep...]")

        partition_loader_train_val = partition_train_test["train"].train_test_split(
            test_size=0.2, seed=preferences.node_shuffle_seed
        )
        train = partition_loader_train_val["train"].to_pandas()
        val = partition_loader_train_val["test"].to_pandas()

        x_train, y_train, _ = prepare_abalone(
            abalone_df=train,
            scaler=preferences.scaler,
        )

        x_val, y_val, _ = prepare_abalone(
            abalone_df=val,
            scaler=preferences.scaler,
        )

        train_dataset = AbaloneDataset(
            x=x_train,
            y=y_train,
        )
        val_dataset = AbaloneDataset(
            x=x_val,
            y=y_val,
        )

        trainloader = DataLoader(train_dataset, batch_size=preferences.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=preferences.batch_size, shuffle=False)

        return FlowerClient(
            trainloader=trainloader, valloader=val_loader, preferences=preferences, partition_id=partition
        ).to_client()
    print("[Preparing data for cross-silo...]")
    train = partition_train_test["train"].to_pandas()
    test = partition_train_test["test"].to_pandas()

    x_train, y_train, _ = prepare_abalone(
        abalone_df=train,
        scaler=preferences.scaler,
    )

    x_test, y_test, _ = prepare_abalone(
        abalone_df=test,
        scaler=preferences.scaler,
    )

    train_dataset = AbaloneDataset(
        x=x_train,
        y=y_train,
    )
    test_dataset = AbaloneDataset(
        x=x_test,
        y=y_test,
    )

    print("Train dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))

    trainloader = DataLoader(train_dataset, batch_size=preferences.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=preferences.batch_size, shuffle=False)

    return FlowerClient(trainloader=trainloader, valloader=test_loader, preferences=preferences).to_client()
