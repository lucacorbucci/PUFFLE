import os

import pandas as pd
import torch
from Client.client import FlowerClient
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from Utils.preferences import Preferences


class CelebaDataset(Dataset):
    """Definition of the dataset used for the Celeba Dataset."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_path: str,
    ) -> None:
        """Initialization of the dataset.

        Args:
        ----
             about the dataset
            image_path (str): path of the images
            transform (torchvision.transforms, optional): Transformation to apply
            to the images. Defaults to None.
        """
        transform = transforms.Compose(
            [
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ],
        )
        smiling_dict = {-1: 0, 1: 1}
        targets = [smiling_dict[item] for item in dataframe["Smiling"]]
        self.targets = targets
        self.sensitive_attributes = dataframe["Male"]
        self.samples = list(dataframe["image_id"])
        self.n_samples = len(dataframe)
        self.transform = transform
        self.image_path = image_path
        self.indexes = range(len(self.samples))

        self.images = [
            Image.open(os.path.join(self.image_path, sample)).convert(
                "RGB",
            )
            for sample in self.samples
        ]

    def __getitem__(self, index: int) ->  tuple[torch.Tensor, int, int]:
        """Returns a sample from the dataset.

        Args:
            idx (_type_): index of the sample we want to retrieve

        Returns
        -------
            _type_: sample we want to retrieve

        """

        img = self.transform(self.images[index])

        return (
            img,
            int(self.sensitive_attributes[index]),
            int(self.targets[index]),
        )

    def __len__(self) -> int:
        """This function returns the size of the dataset.

        Returns
        -------
            int: size of the dataset
        """
        return self.n_samples

def prepare_celeba(partition: pd.DataFrame, preferences: Preferences) -> DataLoader:
    dataset = partition
    train_dataset = CelebaDataset(
        dataframe=dataset, image_path=preferences.image_path)
    trainloader = DataLoader(train_dataset, batch_size=preferences.batch_size, shuffle=True)

    return trainloader

def prepare_celeba_for_cross_silo(preferences: Preferences, partition: pd.DataFrame, partition_id: int) -> DataLoader:
    partition_train_test = partition.train_test_split(test_size=0.2, seed=preferences.seed)
    if preferences.sweep:
        print("[Preparing data for cross-silo for sweep...]")

        partition_loader_train_val = partition_train_test["train"].train_test_split(
            test_size=0.2, seed=preferences.node_shuffle_seed
        )
        train = partition_loader_train_val["train"]
        val = partition_loader_train_val["test"]

        trainloader = prepare_celeba(train, preferences)
        valloader = prepare_celeba(val, preferences)

        return FlowerClient(
            trainloader=trainloader, valloader=valloader, preferences=preferences, partition_id=partition_id
        ).to_client()
    print("[Preparing data for cross-silo...]")

    train = partition_train_test["train"]
    test = partition_train_test["test"]

    trainloader = prepare_celeba(train, preferences)
    testloader = prepare_celeba(test, preferences)

    return FlowerClient(
        trainloader=trainloader, valloader=testloader, preferences=preferences, partition_id=partition_id
    ).to_client()

