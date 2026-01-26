import itertools
import random
from collections import Counter

import numpy as np
import torch
from FederatedDataset.PartitionTypes.iid_partition import IIDPartition
from FederatedDataset.PartitionTypes.non_iid_partition import NonIIDPartition


class UnbalancedPartition:
    def do_partitioning(
        labels: np.ndarray,
        sensitive_features: np.ndarray,
        num_partitions: int,
        total_num_classes: int,
        alpha: int,
        ratio_list: list,
    ) -> list:
        """This function splits a list of labels in num_partitions parts
        considering the sensitive features. If we have N possible sensitive features
        we will create N groups of indexes so that each group will only have
        instances with that sensitive feature.

        Returns:
            list: a list of lists of indexes
        """

        sensitive_features = [item.item() for item in sensitive_features]
        sensitive_features_with_idx = zip(sensitive_features, range(len(labels)))
        # create a dictionary with key = sensitive feature and value = list of indexes
        grouped_sensitive_features = {}
        for sensitive_feature, idx in sensitive_features_with_idx:
            if sensitive_feature not in grouped_sensitive_features:
                grouped_sensitive_features[sensitive_feature] = []
            grouped_sensitive_features[sensitive_feature].append(idx)

        labels = torch.tensor(labels)
        splitted_indexes = []

        for indexes, ratio in zip(grouped_sensitive_features.values(), ratio_list):
            # shuffle the indexes
            random.shuffle(indexes)
            indexes = torch.tensor(indexes)
            current_labels = labels[indexes]
            # call the do_iid_partitioning_with_indexes function from the iid_partition.py file
            splitted_indexes += NonIIDPartition.do_partitioning_with_indexes(
                indexes=indexes,
                labels=current_labels,
                num_partitions=int(num_partitions * ratio),
                alpha=alpha,
            )
        splitted_indexes = list(itertools.chain.from_iterable(zip(*splitted_indexes)))
        return splitted_indexes
