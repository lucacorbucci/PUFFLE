import os

import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from torch.utils.data import Dataset




class CelebaDataset(Dataset):
    """Definition of the dataset used for the Celeba Dataset."""

    def __init__(
        self,
        csv_path: str,
        image_path: str,
        transform: torchvision.transforms = None,
        debug: bool = True,
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
        # self.sensitive_attributes = [smiling_dict[item] for item in dataframe["Gender"].tolist()]
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
            self.sensitive_attributes[index],
            self.targets[index],
        )

    def __len__(self) -> int:
        """This function returns the size of the dataset.

        Returns
        -------
            int: size of the dataset
        """
        return self.n_samples