import random
from typing import Tuple

import numpy as np
import torch
from torchvision import transforms

from puffle.FederatedDataset.Utils.celeba import CelebaDataset
from puffle.FederatedDataset.Utils.dutch import TabularDataset
from puffle.FederatedDataset.Utils.tabular_dataset_utils import (
    dataset_to_numpy,
    load_dutch,
)


class DatasetUtils:
    @staticmethod
    def download_celeba(
        train_csv: str,
        debug: bool,
        test_csv: str = None,
        base_path: str = "../data/celeba",
        # base_path: str = "/mnt/NAS/user/luca.corbucci/data/celeba",
    ) -> Tuple[CelebaDataset, CelebaDataset]:
        """This function downloads the celeba dataset.

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
        train_dataset = CelebaDataset(
            csv_path=f"{base_path}/{train_csv}.csv",
            image_path=f"{base_path}/img_align_celeba",
            transform=transform,
            debug=debug,
        )
        test_dataset = None
        if test_csv:
            test_dataset = CelebaDataset(
                csv_path=f"{base_path}/{test_csv}.csv",
                image_path=f"{base_path}/img_align_celeba",
                transform=transform,
                debug=debug,
            )

        return train_dataset, test_dataset

    @staticmethod
    def download_dataset(
        dataset_name: str,
        train_csv: str = None,
        debug: bool = False,
        test_csv: str = None,
        base_path: str = "../data/celeba",
    ) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
        """This function downloads the required dataset.

        Args:
            dataset_name (str): name of the dataset to download
            train_csv (str): name of the train csv
            test_csv (str): name of the test csv
            base_path (str, optional): base path where the dataset is stored.

        Raises:
            ValueError: if the dataset is not supported

        Returns:
            Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
            the train and test dataset
        """
        if dataset_name == "celeba":
            return DatasetUtils.download_celeba(
                train_csv=train_csv,
                test_csv=test_csv,
                base_path=base_path,
                debug=debug,
            )
        elif dataset_name == "dutch":
            tmp = load_dutch(dataset_path=base_path)
            tmp = dataset_to_numpy(*tmp, num_sensitive_features=1)

            x = tmp[0]
            y = tmp[2]
            z = tmp[1]

            xyz = list(zip(x, y, z))
            random.shuffle(xyz)
            x, y, z = zip(*xyz)
            train_size = int(len(y) * 0.8)

            x_train = np.array(x[:train_size])
            x_test = np.array(x[train_size:])
            y_train = np.array(y[:train_size])
            y_test = np.array(y[train_size:])
            z_train = np.array(z[:train_size])
            z_test = np.array(z[train_size:])

            train_dataset = TabularDataset(
                x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(
                    np.float32
                ),
                z=z_train.astype(np.float32),
                y=y_train.astype(np.float32),
            )

            test_dataset = TabularDataset(
                x=np.hstack((x_test, np.ones((x_test.shape[0], 1)))).astype(np.float32),
                z=z_test.astype(np.float32),
                y=y_test.astype(np.float32),
            )

            return train_dataset, test_dataset
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

    @staticmethod
    def prepare_datasets(
        train_ds: torch.utils.data.Dataset,
        test_ds: torch.utils.data.Dataset,
        batch_size: int,
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Given the train and the test dataset, this function
        returns the train and test dataloader.

        Args:
            train_ds (torch.utils.data.Dataset): train dataset
            test_ds (torch.utils.data.Dataset): test dataset
            batch_size (int):

        Returns:
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
            the train and test dataloader
        """
        kwargs = {"num_workers": 0, "pin_memory": True}

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, **kwargs
        )

        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=True, **kwargs
        )

        return train_loader, test_loader
