import os
from typing import Any

import dill


class StorageManager:
    """This class is used to store the model on disk."""

    @staticmethod
    def write_splitted_dataset(
        dataset_name: str,
        splitted_dataset: list,
        dataset_type: str,
        names: list,
    ) -> None:
        """This function writes the splitted dataset in a pickle file
        and then stores it on disk.

        Args:
        ----
            dataset_name (str): name of the dataset
            splitted_dataset (Any): list of splitted dataset
            dataset_type (str): Type of the dataset. i.e train, test, validation
            names (List[str]): names of the nodes
        """
        for dataset, file_name in zip(splitted_dataset, names):
            filename = f"../data/{dataset_name}/federated_split/{dataset_type}/{file_name}_split"
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            with open(filename, "wb") as file:
                dill.dump(dataset, file)

    @staticmethod
    def write_validation_dataset(
        dataset_name: str,
        dataset: Any,
        dataset_type: str,
    ) -> None:
        """This is used to write the validation dataset on disk.

        Args:
        ----
            dataset_name (str): The name of the dataset
            dataset (Any): The dataset to write
            dataset_type (str): The type of the dataset
        """
        filename = f"../data/{dataset_name}/federated_split/{dataset_type}/{dataset_type}_split"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as file:
            dill.dump(dataset, file)
