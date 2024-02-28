import random

import torch


class PercentagePartition:
    def do_partitioning(
        dataset: torch.utils.data.Dataset,
        percentage_configuration_clusters: dict,
    ) -> dict:
        """_summary_

        Args:
            dataset (torch.utils.data.Dataset): _description_
            num_partitions (int): _description_
            config (Preferences): _description_
            percentage_configuration_clusters (dict): _description_

        Returns:
            dict: _description_
        """
        if not percentage_configuration_clusters:
            raise ValueError("The number of partitions must be greater than 0")

        # We consider the targets from the dataset and we create a
        # dictionary with the possible targets as keys and the
        # corresponding indexes as values. For example: suppose that
        # the we are considering target 0 and that in the dataset only
        # 3 samples have target 0 and they are in position 0, 3 and 10.
        # Then the dictionary will be: {0: [0, 3, 10]}
        # At the same time we store a dictionary with the number of
        # samples for each target.
        targets = dataset.targets
        target_indexes = {}
        target_indexes_sizes = {}
        for target in list({item.item() for item in targets}):
            indexes = [index for index, item in enumerate(targets == target) if item]
            random.shuffle(indexes)
            target_indexes[str(target)] = indexes
            target_indexes_sizes[str(target)] = len(target_indexes[str(target)])

        # Now we consider the percentage configuration for the dataset
        # and for each cluster we consider the percentage configuration
        # for each class and then we get the corresponding percentage
        # of samples for each class.
        # In percentage_split we will store the indexes of the samples
        # that will be assigned to each cluster. For instance if we are considering
        # cluster_0 and we want to assign 80% of label 0 then we will
        # take the first 80% of the indexes of label 0 and we will
        # assign them to cluster_0.
        splitted_indexes = {}
        labels_per_cluster = {}

        for (
            cluster_name,
            percentage_details,
        ) in percentage_configuration_clusters.items():
            splitted_indexes[cluster_name] = []
            labels_per_cluster[cluster_name] = []

            for target, percentage in percentage_details.items():
                selected_indexes = target_indexes[target][
                    : int(percentage / 100 * target_indexes_sizes[target])
                ]
                # We assign the selected indexes and the corresponding
                # labels to the cluster
                splitted_indexes[cluster_name].append(selected_indexes)
                labels_per_cluster[cluster_name] += targets[selected_indexes]

                # We remove from target_indexes the indexes that we have
                # assigned to this cluster. Because we don't want to samle them again
                target_indexes[target] = target_indexes[target][
                    int(percentage / 100 * target_indexes_sizes[target]) :
                ]

        return splitted_indexes, labels_per_cluster
