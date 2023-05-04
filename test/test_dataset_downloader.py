import os
import shutil
from collections import Counter
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from fl_puf.Utils.dataset_utils import DatasetDownloader, TorchVision_FL
from PIL import Image


class TestDatasetDownloader:
    @staticmethod
    def test_do_fl_partitioning() -> None:
        """Test the fl partitioning to be sure that
        data is correctly partitioned among the clients.
        """
        num_images = 100

        # Generate the images
        images = []
        for i in range(num_images):
            # Generate a random numpy array
            arr = np.random.randint(0, 255, size=(28, 28), dtype=np.uint8)

            # Create a PIL image from the numpy array
            img = Image.fromarray(arr)

            # Add the image to the list of images
            images.append(img)
        dummy_sensitive_attribute = torch.randint(0, 2, size=(100,))
        dummy_labels = torch.randint(0, 2, size=(100,))

        # Mock the torch.load() function to return the fake data
        # torch.load = MagicMock(
        #     return_value=(images, dummy_sensitive_attribute, dummy_labels),
        # )

        with patch(
            "torch.load",
            MagicMock(
                return_value=(images, dummy_sensitive_attribute, dummy_labels),
            ),
        ):
            # your code here
            folder = Path("./data/test_data/test_data")
            if folder.exists():
                shutil.rmtree(folder)
            Path.mkdir(folder, parents=True)

            fed_dir = DatasetDownloader.do_fl_partitioning(
                folder,
                pool_size=5,
                alpha=float("inf"),
                num_classes=2,
                val_ratio=0.2,
            )
        # # Check that the data is correctly partitioned
        # # Open the files divided in directories in fed_dir directory
        result = [str(fed_dir) + "/" + str(x[0]) for x in os.listdir(fed_dir)]

        total_data = 0
        sensitive = []
        labels = []

        for folder in result:
            training_data = TorchVision_FL(
                Path(f"{folder}/train.pt"),
            )
            sensitive += training_data.sensitive_features.tolist()
            labels += training_data.targets.tolist()

            validation_data = TorchVision_FL(
                Path(f"{folder}/val.pt"),
            )

            total_data += len(training_data.data) + len(validation_data.data)
            sensitive += validation_data.sensitive_features.tolist()
            labels += validation_data.targets.tolist()

        assert total_data == 100
        assert Counter(sensitive) == Counter(dummy_sensitive_attribute.tolist())
        assert Counter(labels) == Counter(dummy_labels.tolist())
