# ABOUTME: Custom Dataset class for the CelebA image dataset.
# ABOUTME: Provides access to images and sensitive attributes for fairness research.

import os
from typing import Any

import pandas as pd
import torchvision
from PIL import Image
from torch.utils.data import Dataset


class CelebaDataset(Dataset):
    """Definition of the dataset used for the Celeba Dataset."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_path: str,
        transform: Any = None,
        *,
        debug: bool = True,
    ) -> None:
        """
        Initialization of the dataset.

        Args:
            dataframe (pd.DataFrame): The dataframe containing the dataset metadata.
            image_path (str): Path of the images.
            transform (torchvision.transforms, optional): Transformation to apply to the images. Defaults to None.
            debug (bool): Whether to run in debug mode. Defaults to True.

        """
        smiling_dict = {-1: 0, 1: 1}
        targets = [smiling_dict[item] for item in dataframe["Smiling"].tolist()]
        self.targets = targets
        self.sensitive_attributes = dataframe["Male"].tolist()
        self.samples = list(dataframe["image_id"])
        self.n_samples = len(dataframe)
        self.transform = transform
        self.image_path = image_path
        self.debug = debug
        self.indexes = range(len(self.samples))

        if not self.debug:
            self.images = [
                Image.open(os.path.join(self.image_path, sample)).convert(
                    "RGB",
                )
                for sample in self.samples
            ]

    def __getitem__(self, index: int):
        """
        Returns a sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple: The image, sensitive attribute, target, and index.

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
            self.sensitive_attributes[index],
            self.targets[index],
            self.indexes[index],
            index,
        )

    def __len__(self) -> int:
        """
        Return the size of the dataset.

        Returns:
            int: Size of the dataset.

        """
        return self.n_samples
