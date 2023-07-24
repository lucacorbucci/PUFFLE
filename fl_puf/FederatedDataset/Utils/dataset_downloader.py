from typing import Tuple

import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms

from FederatedDataset.Utils.custom_dataset import (
    CelebaDataset,
    CelebaGenderDataset,
    MyDataset,
)


class DatasetDownloader:
    """This class is used to download some useful datasets.

    Raises
    ------
        InvalidDatasetName: If the dataset name is not valid
    """

    @staticmethod
    def download_dataset(dataset_name: str) -> Tuple:
        """Downloads the dataset with the given name.

        Args:
            dataset_name (str): The name of the dataset to download
        Raises:
            InvalidDatasetName: If the dataset name is not valid

        Returns
        -------
            _type_: The downloaded dataset
        """
        if dataset_name == "mnist":
            train_ds, test_ds = DatasetDownloader.download_mnist()
        elif dataset_name == "cifar10":
            train_ds, test_ds = DatasetDownloader.download_cifar10()
        elif dataset_name == "adult":
            train_ds, test_ds = DatasetDownloader.download_adult()
        elif dataset_name == "celeba":
            train_ds, test_ds = DatasetDownloader.download_celeba()
        elif dataset_name == "celeba_sensitive_feature":
            train_ds, test_ds = DatasetDownloader.download_celeba_sensitive_feature()
        elif dataset_name == "celeba_gender":
            train_ds, test_ds = DatasetDownloader.download_celeba_gender()
        elif dataset_name == "fashion_mnist":
            train_ds, test_ds = DatasetDownloader.download_fashion_mnist()
        elif dataset_name == "imaginette":
            train_ds, test_ds = DatasetDownloader.download_imaginette()
        else:
            raise ValueError("Invalid dataset name")
        return train_ds, test_ds

    @staticmethod
    def download_mnist() -> (
        Tuple[
            torchvision.datasets.MNIST,
            torchvision.datasets.MNIST,
        ]
    ):
        """This function downloads the mnist dataset.

        Returns
        -------
            Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
            the train and test dataset
        """
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ],
        )
        mnist_train_ds = torchvision.datasets.MNIST(
            "../data",
            train=True,
            download=True,
            transform=transform,
        )

        mnist_test_ds = torchvision.datasets.MNIST(
            "../data",
            train=False,
            transform=transform,
        )
        return mnist_train_ds, mnist_test_ds

    @staticmethod
    def download_fashion_mnist() -> (
        Tuple[
            torchvision.datasets.FashionMNIST,
            torchvision.datasets.FashionMNIST,
        ]
    ):
        """This function downloads the mnist dataset.

        Returns
        -------
            Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
            the train and test dataset
        """
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))],
        )
        mnist_train_ds = torchvision.datasets.FashionMNIST(
            "../data",
            train=True,
            download=True,
            transform=transform,
        )

        mnist_test_ds = torchvision.datasets.FashionMNIST(
            "../data",
            train=False,
            transform=transform,
        )
        return mnist_train_ds, mnist_test_ds

    @staticmethod
    def download_imaginette() -> (
        Tuple[
            torchvision.datasets.ImageFolder,
            torchvision.datasets.ImageFolder,
        ]
    ):
        """This function downloads the mnist dataset.

        Returns
        -------
            Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
            the train and test dataset
        """
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        )

        mnist_train_ds = torchvision.datasets.ImageFolder(
            "../data/imaginette/train",
            transform=transform,
        )

        mnist_test_ds = torchvision.datasets.ImageFolder(
            "../data/imaginette/val",
            transform=transform,
        )
        return mnist_train_ds, mnist_test_ds

    @staticmethod
    def download_celeba() -> (
        Tuple[
            torchvision.datasets.CelebA,
            torchvision.datasets.CelebA,
        ]
    ):
        """This function downloads the mnist dataset.

        Returns
        -------
            Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
            the train and test dataset
        """
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((64, 64)),
            ],
        )
        train_dataset = CelebaGenderDataset(
            csv_path="../data/celeba/train.csv",
            image_path="../data/celeba/img_align_celeba",
            transform=transform,
        )
        test_dataset = CelebaGenderDataset(
            csv_path="../data/celeba/test.csv",
            image_path="../data/celeba/img_align_celeba",
            transform=transform,
        )
        return train_dataset, test_dataset

    @staticmethod
    def download_celeba_sensitive_feature() -> (
        Tuple[
            torchvision.datasets.CelebA,
            torchvision.datasets.CelebA,
        ]
    ):
        """This function downloads the mnist dataset.

        Returns
        -------
            Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
            the train and test dataset
        """
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((64, 64)),
            ],
        )
        train_dataset = CelebaDataset(
            csv_path="../data/celeba/train.csv",
            image_path="../data/celeba/img_align_celeba",
            transform=transform,
            debug=True,
        )
        test_dataset = CelebaDataset(
            csv_path="../data/celeba/test.csv",
            image_path="../data/celeba/img_align_celeba",
            transform=transform,
            debug=True,
        )
        return train_dataset, test_dataset

    @staticmethod
    def download_celeba_gender() -> (
        Tuple[
            torchvision.datasets.CelebA,
            torchvision.datasets.CelebA,
        ]
    ):
        """This function downloads the mnist dataset.

        Returns
        -------
            Tuple[torchvision.datasets.MNIST, torchvision.datasets.MNIST]:
            the train and test dataset
        """
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(0.5),
                transforms.Resize((64, 64)),
            ],
        )
        train_dataset = CelebaGenderDataset(
            csv_path="../data/celeba/train_gender.csv",
            image_path="../data/celeba/img_align_celeba",
            transform=transform,
        )
        test_dataset = CelebaGenderDataset(
            csv_path="../data/celeba/test_gender.csv",
            image_path="../data/celeba/img_align_celeba",
            transform=transform,
        )
        return train_dataset, test_dataset

    @staticmethod
    def download_cifar10() -> (
        Tuple[
            torchvision.datasets.CIFAR10,
            torchvision.datasets.CIFAR10,
        ]
    ):
        """This function downloads the cifar10 dataset.

        Returns
        -------
            Tuple[torchvision.datasets.CIFAR10, torchvision.datasets.CIFAR10]:
            the train and test dataset
        """
        preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ],
        )

        cifar_train_ds = torchvision.datasets.CIFAR10(
            "../data/files",
            train=True,
            transform=preprocess,
            download=True,
        )

        cifar_test_ds = torchvision.datasets.CIFAR10(
            "../data/files",
            train=False,
            transform=preprocess,
            download=True,
        )
        return cifar_train_ds, cifar_test_ds

    @staticmethod
    def download_adult() -> (
        Tuple[
            torch.utils.data.DataLoader,
            torch.utils.data.DataLoader,
        ]
    ):
        """This function downloads the adult dataset.

        Returns
        -------
            Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
            the train and test dataset
        """
        path_train = "../data/adult/nuovi_dati/train"
        path_test = "../data/adult/nuovi_dati/test"
        train_data = datasets.DatasetFolder(path_train, np.load, ("npy"))
        test_data = datasets.DatasetFolder(path_test, np.load, ("npy"))

        samples = []
        targets = []
        for sample, target in train_data:
            samples.append(sample)
            targets.append(target)

        samples, targets = Utils.shuffle_lists(samples, targets)

        samples_test = []
        targets_test = []
        for sample, target in test_data:
            samples_test.append(sample)
            targets_test.append(target)

        samples_test, targets_test = Utils.shuffle_lists(samples_test, targets_test)

        adult_dataset_train = MyDataset(samples, targets)
        adult_dataset_test = MyDataset(samples_test, targets_test)

        return adult_dataset_train, adult_dataset_test
        return adult_dataset_train, adult_dataset_test
        return adult_dataset_train, adult_dataset_test
        return adult_dataset_train, adult_dataset_test
