import itertools
import random
from collections import Counter

import numpy as np
import torch
from fl_puf.FederatedDataset.PartitionTypes.iid_partition import IIDPartition
from fl_puf.FederatedDataset.PartitionTypes.non_iid_partition import NonIIDPartition


class BalancedAndUnbalanced:
    def do_partitioning(
        labels: np.ndarray,
        sensitive_features: np.ndarray,
        num_partitions: int,
        total_num_classes: int,
        alpha: int,
        ratio_unbalanced: float,
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

        # We create a dictionary with key = (label, sensitive_feature) and value = list of indexes
        # Then we compute the minimum number of instances for each (label, sensitive_feature) pair
        labels_and_sensitive = {}
        for label, sens_feature, index in zip(
            labels, sensitive_features, range(len(labels))
        ):
            if (label.item(), sens_feature) not in labels_and_sensitive:
                labels_and_sensitive[(label.item(), sens_feature)] = []
            labels_and_sensitive[(label.item(), sens_feature)].append(index)

        min_instances = min(
            [len(index_list) for index_list in labels_and_sensitive.values()]
        )
        print(
            "LENGHT: ",
            [len(index_list) for index_list in labels_and_sensitive.values()],
        )

        # Now we want to reduce the number of instances for each (label, sensitive_feature) pair
        # so that we have the same number of instances for each pair. We do this by randomly
        # removing instances from the list of indexes of each pair. Each pair will have at most
        # min_instances instances. The removed instances must be added to
        # another dictionary called removed_instances. This dictionary will be used later to
        # create the unbalanced partitions.

        removed_instances = {}
        for key, index_list in labels_and_sensitive.items():
            removed_instances[key] = []
            while len(index_list) > min_instances:
                removed_instances[key].append(
                    index_list.pop(random.randrange(len(index_list)))
                )

        # Now we create a list with all the indexes of the instances that we have not removed
        # from the labels_and_sensitive dictionary. We will use this list to create the balanced
        # partition using a IID distribution
        balanced_indexes = np.array(
            list(itertools.chain.from_iterable(labels_and_sensitive.values()))
        )
        balanced_partition = IIDPartition.do_iid_partitioning_with_indexes(
            indexes=balanced_indexes, num_partitions=num_partitions
        )

        # Now we can get the ratio% of the balanced partition and unbalanced them
        # using the removed_instances dictionary
        ratio_to_be_unbalanced = int(len(balanced_partition) * ratio_unbalanced)

        to_be_unbalanced = balanced_partition[:ratio_to_be_unbalanced]
        balanced_partition = balanced_partition[ratio_to_be_unbalanced:]

        value_to_unbalance = (1, 1)
        opposite_value = (0, 1)
        # Now we take the to_be_unbalanced partition, we remove 2/3 of the instances with
        # the (label, sensitive value) pair equal to value_to_unbalance and then we add
        # the same amount of instances taken from removed_instances with different
        # (label, sensitive value) pair.
        new_to_be_unbalanced = []
        for indexes in to_be_unbalanced:
            # remove 2/3 of the instances with the (label, sensitive value) pair equal to value_to_unbalance
            indexes_to_remove = []
            for index in indexes:
                if (
                    labels[index].item(),
                    sensitive_features[index],
                ) == value_to_unbalance:
                    indexes_to_remove.append(index)
            indexes_to_remove = indexes_to_remove[: int(len(indexes_to_remove) * 2 / 3)]
            removed_indexes = len(indexes_to_remove)
            indexes = np.array(
                [index for index in indexes if index not in indexes_to_remove]
            )

            # add the same amount of instances taken from removed_instances with different
            # (label, sensitive value) pair
            while removed_indexes > 0:
                for key, values in removed_instances.items():
                    if len(values) > 0 and key == opposite_value:
                        indexes = np.append(indexes, values.pop())
                        removed_indexes -= 1

            new_to_be_unbalanced.append(indexes)
        total = 0
        for indexes in new_to_be_unbalanced:
            total += len(indexes)
        for indexes in balanced_partition:
            total += len(indexes)

        print(f"We're using {total} instances")
        print(
            f"Len Balanced {len(balanced_partition)} Len Unbalanced {len(new_to_be_unbalanced)}"
        )

        # We have to sort the indexes so that each time we sample the nodes
        # we have some nodes from unbalanced and some from balanced
        splitted_indexes = []
        ratio_splitted = int(10 * ratio_unbalanced)

        for index in range(len(balanced_partition) + len(new_to_be_unbalanced)):
            # The first ratio_splitted % of the nodes will be underrepresented in each batch of nodes
            if index % 10 < ratio_splitted:
                splitted_indexes.append(new_to_be_unbalanced.pop())
            else:
                splitted_indexes.append(balanced_partition.pop())

        # return the merge of the balanced partition and the unbalanced partition
        return splitted_indexes
