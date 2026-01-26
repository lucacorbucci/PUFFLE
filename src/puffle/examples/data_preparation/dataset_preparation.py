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
    base_path: str = "../../data/celeba",
    sweep: bool = False,
    validation_seed: int = 42,
    seed: int = 490,
) -> tuple[CelebaDataset, CelebaDataset]:
    """
    This function downloads the celeba dataset.

    Args:
        train_csv (str): name of the train_csv
        test_csv (str): name of the test csv
        base_path (str, optional): base path where the dataset is stored.
        Defaults to "/mnt/NAS/user/luca.corbucci/data/celeba".

    Returns:
        Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
        the train and test dataset

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
    tmp = load_dutch(dataset_path=base_path)
    tmp = dataset_to_numpy(*tmp, num_sensitive_features=1)

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
        x_train, y_train, z_train = zip(*random.sample(list(zip(x_train, y_train, z_train, strict=False)), len(x_train)), strict=False)

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
    sensitive_features_last: bool = True,
):
    """
    Args:
    _df: pandas dataframe
    _feature_cols: list of feature column names
    _metadata: dictionary with metadata
    num_sensitive_features: number of sensitive features to use
    sensitive_features_last: if True, then sensitive features are encoded as last columns

    """
    # transform features to 1-hot
    _X = _df[_feature_cols]
    # take sensitive features separately
    num_sensitive_features = min(num_sensitive_features, len(_metadata["protected_atts"]))
    _Z = _X[_metadata["protected_atts"][:num_sensitive_features]]
    _X = _X.drop(columns=_metadata["protected_atts"][:num_sensitive_features])
    # 1-hot encode and scale features
    dummy_cols = _metadata.get("dummy_cols")
    _X2 = pd.get_dummies(_X, columns=dummy_cols, drop_first=False)
    esc = MinMaxScaler()
    _X = esc.fit_transform(_X2)

    # current implementation assumes each sensitive feature is binary
    for i, tmp in enumerate(_metadata["protected_atts"][:num_sensitive_features]):
        if len(_Z[tmp].unique()) != 2:
            msg = "Sensitive feature is not binary!"
            raise ValueError(msg)

    # 1-hot sensitive features, (optionally) swap ordering so privileged class feature == 1 is always last, preceded by the corresponding unprivileged feature
    _Z2 = pd.get_dummies(_Z, columns=_Z.columns, drop_first=False)
    if sensitive_features_last:
        for i, tmp in enumerate(_Z.columns):
            if _metadata["protected_att_values"][i] not in _Z[tmp].unique():
                msg = "Protected attribute value not found in data!"
                raise ValueError(msg)
            if not np.allclose(float(_metadata["protected_att_values"][i]), 0):
                # swap columns
                _Z2.iloc[:, [2 * i, 2 * i + 1]] = _Z2.iloc[:, [2 * i + 1, 2 * i]]
    # change booleans to floats
    _y = _df[_metadata["target_variable"]].values
    return _X, np.array([sv[0] for sv in _Z.values]), _y
