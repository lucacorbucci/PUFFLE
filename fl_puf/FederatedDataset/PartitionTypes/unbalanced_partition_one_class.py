import itertools
import random
from collections import Counter

import numpy as np
import torch
from FederatedDataset.PartitionTypes.iid_partition import IIDPartition
from FederatedDataset.PartitionTypes.non_iid_partition import NonIIDPartition


class UnbalancedPartitionOneClass:
    def do_partitioning(
        labels: np.ndarray,
        sensitive_features: np.ndarray,
        num_partitions: int,
        total_num_classes: int,
        alpha: int,
        ratio: float,
    ) -> list:
        """

        Returns:
            list: a list of lists of indexes
        """

        sensitive_features = [item.item() for item in sensitive_features]
        features_and_labels = zip(sensitive_features, labels, range(len(labels)))

        # Number of nodes that will have 3 combinations
        num_nodes_with_2_combinations = int(ratio * num_partitions)
        num_nodes_with_4_combinations = num_partitions - num_nodes_with_2_combinations

        indexes = range(len(labels))
        current_labels = [item.item() for item in labels]

        indexes_and_labels = dict(zip(indexes, current_labels))
        indexes_and_sensitive_features = dict(zip(indexes, sensitive_features))

        splitted_indexes = []
        indexes = torch.tensor(indexes)
        labels = torch.tensor(labels)
        # splitted_indexes += NonIIDPartition.do_partitioning_with_indexes(indexes=indexes, labels=labels, num_partitions=num_partitions, alpha=alpha)
        splitted_indexes += IIDPartition.do_iid_partitioning_with_indexes(
            indexes=indexes, num_partitions=num_partitions
        )

        new_index_list = []
        for index_list in splitted_indexes:
            current_list = []
            if num_nodes_with_2_combinations > 0:
                for item in index_list:
                    item = item.item() if isinstance(item, torch.Tensor) else item
                    # I want to have only 2 combinations: 1,-1 and 0,-1 in num_nodes_with_2_combinations
                    if (
                        indexes_and_labels[item] == 1
                        and indexes_and_sensitive_features[item] == 1
                    ) or (
                        indexes_and_labels[item] == 0
                        and indexes_and_sensitive_features[item] == 1
                    ):
                        continue
                    else:
                        current_list.append(item)

                new_index_list.append(current_list)
                num_nodes_with_2_combinations -= 1
            else:
                new_index_list.append(index_list)

        random.seed(42)
        # splitted_indexes = list(itertools.chain.from_iterable(zip(*splitted_indexes)))
        random.shuffle(new_index_list)
        return new_index_list
