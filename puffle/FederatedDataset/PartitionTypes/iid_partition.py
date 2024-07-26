from collections import Counter

import numpy as np
import torch


class IIDPartition:
    def do_partitioning(
        dataset: torch.utils.data.Dataset,
        num_partitions: int,
        total_num_classes: int,
    ) -> dict:
        labels = dataset.targets
        # print(Counter([item.item() for item in labels]))

        num_labels = len(set([item.item() for item in labels]))
        idx = torch.tensor(list(range(len(labels))))
        # shuffle the indexes
        idx = idx[torch.randperm(len(idx))]

        # split the indexes into num_partitions
        splitted_indexes = np.array_split(idx, num_partitions)
        splitted_labels = [labels[index_list] for index_list in splitted_indexes]

        splitted_indexes_dict = {
            f"cluster_{index}": item for index, item in enumerate(splitted_indexes)
        }
        splitted_labels_dict = {
            f"cluster_{index}": item for index, item in enumerate(splitted_labels)
        }
        return splitted_indexes_dict, splitted_labels_dict

    def do_iid_partitioning_with_indexes(
        indexes: np.array,
        num_partitions: int,
    ) -> np.array:
        """This function splits a list of indexes in N parts.
        First of all the list is shuffled and then it is splitted in N parts.

        Args:
            indexes (np.array): the list of indexes to be splitted
            num_partitions (int): the number of partitions

        Returns:
            np.array: the list of splitted indexes
        """
        idx = indexes[torch.randperm(len(indexes))]
        return np.array_split(idx, num_partitions)
