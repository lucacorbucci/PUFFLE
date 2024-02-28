import os
from typing import Any, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    """Definition of generic custom dataset."""

    def __init__(self, samples: list, targets: list, transform) -> None:
        self.data: list = samples
        self.targets = torch.tensor(targets)
        self.indices = np.asarray(range(len(self.targets)))
        self.transform = transform

    def __len__(self) -> int:
        """This function returns the size of the dataset.

        Returns
        -------
            _type_: size of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        """Returns a sample from the dataset.

        Args:
            idx (_type_): index of the sample we want to retrieve

        Returns
        -------
            _type_: sample we want to retrieve
        """
        img, target = self.data[idx], self.targets[idx]
        if self.transform is not None:
            img = self.transform(img)

        return img, target


class MyDatasetWithCSV(Dataset):
    """Definition of generic custom dataset."""

    def __init__(
        self,
        targets: list,
        image_path: str,
        image_ids: list,
        sensitive_features: list,
        transform: torchvision.transforms = None,
    ) -> None:
        self.data: list = image_ids
        self.targets = torch.tensor(list(targets))
        self.classes = torch.tensor(list(targets))
        self.indices = np.asarray(range(len(self.targets)))
        self.samples = image_ids
        self.n_samples = len(self.samples)
        self.transform = transform
        self.image_path = image_path
        self.sensitive_features = sensitive_features

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a sample from the dataset.

        Args:
            idx (_type_): index of the sample we want to retrieve

        Returns
        -------
            _type_: sample we want to retrieve

        """
        img = Image.open(os.path.join(self.image_path, self.samples[index])).convert(
            "RGB",
        )
        if self.transform:
            img = self.transform(img)

        return transforms.functional.to_tensor(img), self.targets[index]

    def __len__(self) -> int:
        """This function returns the size of the dataset.

        Returns
        -------
            int: size of the dataset
        """
        return self.n_samples


class CelebaGenderDataset(Dataset):
    """Definition of the dataset used for the Celeba Dataset."""

    def __init__(
        self,
        csv_path: str,
        image_path: str,
        transform: torchvision.transforms = None,
    ) -> None:
        """Initialization of the dataset.

        Args:
        ----
            csv_path (str): path of the csv file with all the information
             about the dataset
            image_path (str): path of the images
            transform (torchvision.transforms, optional): Transformation to apply
            to the images. Defaults to None.
        """
        dataframe = pd.read_csv(csv_path)
        self.targets = dataframe["Target"]
        self.classes = dataframe["Target"]
        self.samples = list(dataframe["image_id"])
        self.n_samples = len(dataframe)
        self.transform = transform
        self.image_path = image_path

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a sample from the dataset.

        Args:
            idx (_type_): index of the sample we want to retrieve

        Returns
        -------
            _type_: sample we want to retrieve

        """
        img = Image.open(os.path.join(self.image_path, self.samples[index])).convert(
            "RGB",
        )
        if self.transform:
            img = self.transform(img)

        return transforms.functional.to_tensor(img), self.targets[index]

    def __len__(self) -> int:
        """This function returns the size of the dataset.

        Returns
        -------
            int: size of the dataset
        """
        return self.n_samples


class CelebaDataset(Dataset):
    """Definition of the dataset used for the Celeba Dataset."""

    def __init__(
        self,
        csv_path: str,
        image_path: str,
        transform: torchvision.transforms = None,
        debug: bool = False,
    ) -> None:
        """Initialization of the dataset.

        Args:
        ----
            csv_path (str): path of the csv file with all the information
             about the dataset
            image_path (str): path of the images
            transform (torchvision.transforms, optional): Transformation to apply
            to the images. Defaults to None.
        """
        dataframe = pd.read_csv(csv_path)

        smiling_dict = {-1: 0, 1: 1}
        targets = [smiling_dict[item] for item in dataframe["Smiling"].tolist()]

        self.targets = targets
        self.classes = targets

        self.sensitive_features = dataframe["Male"].tolist()
        self.samples = list(dataframe["image_id"])
        self.n_samples = len(dataframe)
        self.transform = transform
        self.image_path = image_path
        self.debug = debug
        if not self.debug:
            self.images = [
                Image.open(os.path.join(self.image_path, sample)).convert(
                    "RGB",
                )
                for sample in self.samples
            ]

    def __getitem__(self, index: int):
        """Returns a sample from the dataset.

        Args:
            idx (_type_): index of the sample we want to retrieve

        Returns
        -------
            _type_: sample we want to retrieve

        """
        if self.debug:
            img = Image.open(
                os.path.join(self.image_path, self.samples[index])
            ).convert(
                "RGB",
            )
        else:
            img = self.images[index]

        if self.transform:
            img = self.transform(img)

        return (
            img,
            self.sensitive_feature[index],
            self.targets[index],
        )

    def __len__(self) -> int:
        """This function returns the size of the dataset.

        Returns
        -------
            int: size of the dataset
        """
        return self.n_samples
