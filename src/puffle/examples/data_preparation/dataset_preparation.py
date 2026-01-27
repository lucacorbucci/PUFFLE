# ABOUTME: Utility functions for loading and preparing CelebA and Dutch Census datasets.
# ABOUTME: Handles data splitting, normalization, and sensitive attribute processing.

import random

import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torchvision import transforms

from puffle.examples.data_preparation.celeba import CelebaDataset


def prepare_celeba_centralised(
    debug: bool = True,
    train_csv: str = "train.csv",
    base_path: str = "./data",
    sweep: bool = False,
    validation_seed: int = 42,
    seed: int = 490,
) -> tuple[CelebaDataset, CelebaDataset, CelebaDataset]:
    """
    Download and prepare the CelebA dataset.

    Args:
        debug (bool): Whether to run in debug mode. Defaults to True.
        train_csv (str): Filename of the training CSV. Defaults to "train.csv".
        base_path (str, optional): Base path where the dataset is stored.
        sweep (bool): Whether to perform a hyperparameter sweep split. Defaults to False.
        validation_seed (int): Seed for validation split. Defaults to 42.
        seed (int): Global random seed. Defaults to 490.

    Returns:
        tuple: (train_dataset, test_dataset, val_dataset)

    """
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ],
    )
    dataframe = pd.read_csv(f"{base_path}/{train_csv}")

    dataframe_test = dataframe.sample(frac=0.2, random_state=seed)
    dataframe.drop(dataframe_test.index)

    if sweep:
        random.seed(validation_seed)
        dataframe_val = dataframe.sample(frac=0.2, random_state=validation_seed)
        dataframe.drop(dataframe_val.index)
    else:
        dataframe_val = None

    train_dataset = CelebaDataset(
        dataframe=dataframe,
        image_path=f"{base_path}/img_align_celeba",
        transform=transform,
        debug=debug,
    )

    test_dataset = CelebaDataset(
        dataframe=dataframe_test,
        image_path=f"{base_path}/img_align_celeba",
        transform=transform,
        debug=debug,
    )

    if dataframe_val is not None:
        val_dataset = CelebaDataset(
            dataframe=dataframe_val,
            image_path=f"{base_path}/img_align_celeba",
            transform=transform,
            debug=debug,
        )
    return train_dataset, test_dataset, val_dataset


class TabularDataset(Dataset):
    def __init__(self, x, z, y):
        """
        Initialize the custom dataset with x (features), z (sensitive values), and y (targets).

        Args:
        x (list of tensors): List of input feature tensors.
        z (list): List of sensitive values.
        y (list): List of target values.

        """
        self.samples = x
        self.sensitive_features = z
        self.targets = y
        self.indexes = range(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single data point from the dataset.

        Args:
        idx (int): Index to retrieve the data point.

        Returns:
        sample (dict): A dictionary containing 'x', 'z', and 'y'.

        """
        x_sample = self.samples[idx]
        z_sample = self.sensitive_features[idx]
        y_sample = self.targets[idx]

        return x_sample, z_sample, y_sample, self.indexes[idx], idx


def prepare_dutch(base_path, sweep, validation_seed=None):
    df, cols, meta = load_dutch(dataset_path=base_path)
    tmp = dataset_to_numpy(df, cols, meta, num_sensitive_features=1)

    x = tmp[0]
    y = tmp[2]
    z = tmp[1]

    xyz = list(zip(x, y, z, strict=False))
    random.shuffle(xyz)
    x, y, z = zip(*xyz, strict=False)
    train_size = int(len(y) * 0.8)

    x_train = np.array(x[:train_size])
    x_test = np.array(x[train_size:])
    y_train = np.array(y[:train_size])
    y_test = np.array(y[train_size:])
    z_train = np.array(z[:train_size])
    z_test = np.array(z[train_size:])

    if sweep:
        random.seed(validation_seed)
        # shuffle the data
        x_train, y_train, z_train = zip(
            *random.sample(
                list(zip(x_train, y_train, z_train, strict=False)), len(x_train)
            ),
            strict=False,
        )

        val_size = int(len(x_train) * 0.2)

        x_val = np.array(x_train[-val_size:])
        x_train = np.array(x_train[:-val_size])

        y_val = np.array(y_train[-val_size:])
        y_train = np.array(y_train[:-val_size])

        z_val = np.array(z_train[-val_size:])
        z_train = np.array(z_train[:-val_size])

        val_dataset = TabularDataset(
            x=np.hstack((x_val, np.ones((x_val.shape[0], 1)))).astype(np.float32),
            z=z_val.astype(np.float32),
            y=y_val.astype(np.float32),
        )
    else:
        val_dataset = None

    train_dataset = TabularDataset(
        x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(np.float32),
        z=z_train.astype(np.float32),
        y=y_train.astype(np.float32),
    )

    test_dataset = TabularDataset(
        x=np.hstack((x_test, np.ones((x_test.shape[0], 1)))).astype(np.float32),
        z=z_test.astype(np.float32),
        y=y_test.astype(np.float32),
    )

    return train_dataset, test_dataset, val_dataset


def load_dutch(dataset_path):
    data = arff.loadarff(dataset_path + "dutch_census.arff")
    dutch_df = pd.DataFrame(data[0]).astype("int32")

    dutch_df["sex_binary"] = np.where(dutch_df["sex"] == 1, 1, 0)
    dutch_df["occupation_binary"] = np.where(dutch_df["occupation"] >= 300, 1, 0)

    del dutch_df["sex"]
    del dutch_df["occupation"]

    dutch_df_feature_columns = [
        "age",
        "household_position",
        "household_size",
        "prev_residence_place",
        "citizenship",
        "country_birth",
        "edu_level",
        "economic_status",
        "cur_eco_activity",
        "Marital_status",
        "sex_binary",
    ]

    metadata_dutch = {
        "name": "Dutch census",
        "code": ["DU1"],
        "protected_atts": ["sex_binary"],
        "protected_att_values": [0],
        "protected_att_descriptions": ["Gender = Female"],
        "target_variable": "occupation_binary",
    }

    return dutch_df, dutch_df_feature_columns, metadata_dutch


def dataset_to_numpy(
    _df,
    _feature_cols: list,
    _metadata: dict,
    num_sensitive_features: int = 1,
    *,
    sensitive_features_last: bool = True,
):
    """
    Convert a dataframe to numpy arrays for features, sensitive attributes, and targets.

    Args:
        _df (pd.DataFrame): Input dataframe.
        _feature_cols (list): List of feature column names.
        _metadata (dict): Metadata containing protected attribute information.
        num_sensitive_features (int): Number of sensitive features to extract.
        sensitive_features_last (bool, optional): Whether to place sensitive features last. Defaults to True.

    """
    # transform features to 1-hot
    x_raw = _df[_feature_cols]
    # take sensitive features separately
    num_sensitive_features = min(
        num_sensitive_features, len(_metadata["protected_atts"])
    )
    z_raw = x_raw[_metadata["protected_atts"][:num_sensitive_features]]
    x_raw = x_raw.drop(columns=_metadata["protected_atts"][:num_sensitive_features])
    # 1-hot encode and scale features
    dummy_cols = _metadata.get("dummy_cols")
    x_dummies = pd.get_dummies(x_raw, columns=dummy_cols, drop_first=False)
    esc = MinMaxScaler()
    x_scaled = esc.fit_transform(x_dummies)

    # current implementation assumes each sensitive feature is binary
    for _i, tmp in enumerate(_metadata["protected_atts"][:num_sensitive_features]):
        if len(z_raw[tmp].unique()) != 2:
            msg = "Sensitive feature is not binary!"
            raise ValueError(msg)

    # 1-hot sensitive features, (optionally) swap ordering
    z_dummies = pd.get_dummies(z_raw, columns=z_raw.columns, drop_first=False)
    if sensitive_features_last:
        for i, tmp in enumerate(z_raw.columns):
            if _metadata["protected_att_values"][i] not in z_raw[tmp].unique():
                msg = "Protected attribute value not found in data!"
                raise ValueError(msg)
            if not np.allclose(float(_metadata["protected_att_values"][i]), 0):
                # swap columns
                z_dummies.iloc[:, [2 * i, 2 * i + 1]] = z_dummies.iloc[
                    :, [2 * i + 1, 2 * i]
                ]
    # change booleans to floats
    y_values = _df[_metadata["target_variable"]].to_numpy()
    return x_scaled, np.array([sv[0] for sv in z_raw.to_numpy()]), y_values
