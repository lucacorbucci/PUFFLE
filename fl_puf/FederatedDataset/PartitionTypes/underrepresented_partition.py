import itertools
import random
from collections import Counter

import numpy as np
import torch
from fl_puf.FederatedDataset.PartitionTypes.iid_partition import IIDPartition
from fl_puf.FederatedDataset.PartitionTypes.non_iid_partition import NonIIDPartition


class UnderrepresentedPartition:
    def do_partitioning(
        labels: np.ndarray,
        sensitive_features: np.ndarray,
        num_partitions: int,
        total_num_classes: int,
        alpha: int,
        ratio: float,
    ) -> list:
        """
        # I want to be able to create a group (for instance 30% of the nodes)
        #  with only 2 combinations: 1,1 and 0,1
        # and the rest of the nodes (70%) with 2 combinations: 1,-1 and 0,-1.
        Returns:
            list: a list of lists of indexes
        """

        random.seed(42)
        sensitive_features = [item.item() for item in sensitive_features]

        # Number of nodes that will have 3 combinations
        underrepresented_nodes = int(ratio * num_partitions)
        represented_nodes = num_partitions - underrepresented_nodes

        indexes = range(len(labels))
        current_labels = [item.item() for item in labels]

        underrepresented_indexes = []
        underrepresented_labels = []
        represented_indexes = []
        represented_labels = []
        for index, sens_feature, label in zip(
            indexes, sensitive_features, current_labels
        ):
            if (label == 1 and sens_feature == 1) or (label == 0 and sens_feature == 1):
                underrepresented_indexes.append(index)
                underrepresented_labels.append(label)
            else:
                represented_indexes.append(index)
                represented_labels.append(label)

        splitted_indexes_underrepresented = []
        splitted_indexes_represented = []

        splitted_indexes_underrepresented += (
            NonIIDPartition.do_partitioning_with_indexes(
                indexes=torch.tensor(underrepresented_indexes),
                labels=torch.tensor(underrepresented_labels),
                num_partitions=underrepresented_nodes,
                alpha=alpha,
            )
        )
        splitted_indexes_represented += NonIIDPartition.do_partitioning_with_indexes(
            indexes=torch.tensor(represented_indexes),
            labels=torch.tensor(represented_labels),
            num_partitions=represented_nodes,
            alpha=alpha,
        )

        splitted_indexes = []
        ratio_splitted = int(10 * ratio)
        for index in range(
            len(splitted_indexes_represented) + len(splitted_indexes_underrepresented)
        ):
            # The first ratio_splitted % of the nodes will be underrepresented in each batch of nodes
            if index % 10 < ratio_splitted:
                splitted_indexes.append(splitted_indexes_underrepresented.pop())
            else:
                splitted_indexes.append(splitted_indexes_represented.pop())
        return splitted_indexes
