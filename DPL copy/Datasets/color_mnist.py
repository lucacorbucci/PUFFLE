import os

import numpy as np
import torch
from PIL import Image

# import torchvision.datasets.utils as dataset_utils
# from torchvision.datasets.utils import makedir_exist_ok

__all__ = ["ColorMNIST"]

import torch

torch.manual_seed(42)


def color_grayscale_arr(arr, red=True):
    """Converts grayscale image to either red or green"""
    assert arr.ndim == 2
    dtype = arr.dtype
    h, w = arr.shape
    arr = np.reshape(arr, [h, w, 1])
    if red:
        arr = np.concatenate([arr, np.zeros((h, w, 2), dtype=dtype)], axis=2)
    else:
        arr = np.concatenate(
            [np.zeros((h, w, 1), dtype=dtype), arr, np.zeros((h, w, 1), dtype=dtype)],
            axis=2,
        )
    return arr


class ColoredMNIST(datasets.VisionDataset):
    """
    Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

    Args:
      root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
      env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
      transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
    """

    def __init__(
        self, dataset, split, root="./data", transform=None, target_transform=None
    ):
        super(ColoredMNIST, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.prepare_colored_mnist(
            dataset=dataset, dataset_name="ColorMnist", split=split
        )

        self.data_label_tuples = torch.load(
            os.path.join(self.root, "ColoredMNIST", split) + ".pt"
        )

        self.images = []
        self.labels = []
        self.colors = []
        self.targets = []

        for image, color, label in self.data_label_tuples:
            self.images.append(image)
            self.labels.append(label)
            self.colors.append(color)
            self.targets.append(label)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, color, target = self.data_label_tuples[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, color, target

    def __len__(self):
        return len(self.data_label_tuples)

    def prepare_colored_mnist(self, dataset, dataset_name, split):
        colored_mnist_dir = os.path.join(self.root, "ColoredMNIST")

        if (
            os.path.exists(os.path.join(colored_mnist_dir, "public.pt"))
            and os.path.exists(os.path.join(colored_mnist_dir, "test.pt"))
            and os.path.exists(os.path.join(colored_mnist_dir, "train.pt"))
        ):
            print("Colored MNIST dataset already exists")
            return

        print("Preparing Colored MNIST")
        set_ = []

        for idx, (im, label) in enumerate(dataset):
            if idx % 10000 == 0:
                print(f"Converting image {idx}/{len(dataset)}")
            im_array = np.array(im)

            if split == "train":
                color_red = np.random.binomial(1, 0.95)
            elif split == "test" or split == "public":
                color_red = np.random.binomial(1, 0.5)

            colored_arr = color_grayscale_arr(im_array, red=color_red)

            if split == "test":
                if idx < 5000:
                    set_.append((Image.fromarray(colored_arr), color_red, label))
            elif split == "public":
                if idx > 5000:
                    set_.append((Image.fromarray(colored_arr), color_red, label))
            else:
                set_.append((Image.fromarray(colored_arr), color_red, label))

        torch.save(set_, os.path.join(colored_mnist_dir, f"{split}.pt"))
