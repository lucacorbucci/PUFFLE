import os
from typing import Any

import torch
import torchvision
from Client.client import FlowerClient
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from Utils.preferences import Preferences


class ImageDataset(Dataset):
    """
    Custom Dataset class that handles images, labels, and sensitive attributes."""

    def __init__(self, data: Any, transform: Any) -> None:
        """
        Initializes the ImageDataset with data and optional transforms.

        Stores the data and transform for image loading and preprocessing.

        Args:
            data (Any): Dataset or list containing images, labels, and sensitive attributes.
            transform (Any): Optional transform to apply to images (e.g., ToTensor, Normalize).

        Returns:
            None
        """
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.

        Args:
            None

        Returns:
            int: Size of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        """
        Retrieves a single sample from the dataset by index.

        Applies transform to image if provided; uses default sensitive_attribute=-1 if not present.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, int, int]: (transformed image tensor, sensitive attribute, label)
        """
        # Get the example at the given index
        example = self.data[idx]

        # Extract image, label, and sensitive attribute
        image = example["image"]
        label = example["label"]
        sensitive_attribute = example.get("sensitive_attribute", -1)

        # Apply transforms to the image
        if self.transform:
            image = self.transform(image)

        return image, sensitive_attribute, label


def download_mnist(data_root: str = "../data/") -> Any:
    """
    Downloads MNIST dataset and saves images as PNG files in class subdirectories.

    Combines train/test sets, creates ImageFolder-compatible structure under data_root/MNIST/train/.

    Args:
        data_root (str, optional): Directory to store the dataset. Defaults to "../data/".

    Returns:
        Any: The full concatenated PyTorch dataset.

    Raises:
        OSError: If directory creation or file saving fails.
    """
    print("Starting download of MNIST dataset...")
    # Download the training and testing datasets
    transformer = transforms.ToTensor()
    mnist_train = torchvision.datasets.MNIST("../data", train=True, download=True, transform=transformer)
    mnist_test = torchvision.datasets.MNIST("../data", train=False, download=True, transform=transformer)

    # Combine training and testing data for a complete dataset
    full_dataset = torch.utils.data.ConcatDataset([mnist_train, mnist_test])

    # Create root directory for saving images
    save_root = os.path.join(data_root, "MNIST/train/")
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    # Create a subfolder for each digit class
    for i in range(10):
        class_dir = os.path.join(save_root, str(i))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

    # Iterate through the full dataset and save images
    for i, (image_tensor, label) in enumerate(full_dataset):  # type: ignore
        # Convert the PyTorch tensor to a PIL Image
        # The image tensor is 1x28x28, so we need to squeeze it to 28x28
        image = Image.fromarray((image_tensor.squeeze() * 255).numpy().astype("uint8"))

        # Define the save path
        save_path = os.path.join(save_root, str(label), f"{label}_{i}.png")

        # Save the image
        image.save(save_path)

        # Print progress every 1000 images
        if (i + 1) % 10000 == 0:
            print(f"Saved {i + 1} images...")

    return full_dataset


def prepare_mnist(partition: Any, preferences: Preferences) -> DataLoader:
    """
    Prepares MNIST partition data into an ImageDataset and DataLoader.

    Applies ToTensor and Normalize transforms; shuffles for training.

    Args:
        partition (Any): MNIST data partition.
        preferences (Preferences): Configuration including batch_size.

    Returns:
        DataLoader: Configured DataLoader for the partition.
    """
    train = partition

    train_dataset = ImageDataset(
        train, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    )
    trainloader = DataLoader(train_dataset, batch_size=preferences.batch_size, shuffle=True)

    return trainloader


def prepare_mnist_for_cross_silo(preferences: Preferences, partition: Any, partition_id: int) -> Any:
    """
    Prepares MNIST data for cross-silo federated learning from a partition.

    Splits into train/test (20% test), optionally train/val for sweep; uses prepare_mnist for loaders, creates FlowerClient.

    Args:
        preferences (Preferences): FL configuration including batch_size, seed.
        partition (Any): Data partition for this client.
        partition_id (int): Client partition ID (unused in implementation).

    Returns:
        Any: FlowerClient instance wrapped as .to_client().

    Raises:
        ValueError: If data splitting or processing fails.
    """
    partition_train_test = partition.train_test_split(test_size=0.2, seed=preferences.seed)
    if preferences.sweep:
        print("[Preparing data for cross-silo for sweep...]")

        partition_loader_train_val = partition_train_test["train"].train_test_split(
            test_size=0.2, seed=preferences.node_shuffle_seed
        )
        train = partition_loader_train_val["train"]
        val = partition_loader_train_val["test"]

        trainloader = prepare_mnist(train, preferences)
        valloader = prepare_mnist(val, preferences)

        return FlowerClient(
            trainloader=trainloader, valloader=valloader, preferences=preferences, partition_id=partition_id
        ).to_client()
    print("[Preparing data for cross-silo...]")

    train = partition_train_test["train"]
    test = partition_train_test["test"]

    trainloader = prepare_mnist(train, preferences)
    testloader = prepare_mnist(test, preferences)

    return FlowerClient(
        trainloader=trainloader, valloader=testloader, preferences=preferences, partition_id=partition_id
    ).to_client()
