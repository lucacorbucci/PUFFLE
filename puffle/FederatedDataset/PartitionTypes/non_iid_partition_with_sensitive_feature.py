from collections import Counter

import numpy as np
import torch


class NonIIDPartitionWithSensitiveFeature:
    def do_partitioning(
        dataset: torch.utils.data.Dataset,
        num_partitions: int,
        total_num_classes: int,
        alpha=1000000,
    ) -> dict:
        if not alpha:
            raise ValueError("Alpha must be a positive number")
        labels = dataset.targets
        sensitive_features = dataset.sensitive_features
        indexes = np.array(list(range(len(labels))))
        data = dataset.data if hasattr(dataset, "data") else dataset.samples

        # num_labels = len(set([item.item() for item in labels]))
        idx = torch.tensor(list(range(len(labels))))

        index_per_label = {}
        for index, label in zip(idx, labels):
            if isinstance(label, torch.Tensor):
                label = label.item()
            if label not in index_per_label:
                index_per_label[label] = []
            index_per_label[label].append(index)

        # in list labels we have the labels of this dataset
        list_labels = {
            item.item() if isinstance(item, torch.Tensor) else item for item in labels
        }
        list_sensitive_features = {
            item.item() if isinstance(item, torch.Tensor) else item
            for item in sensitive_features
        }
        labels_and_sensitive_feature = []
        for label in list_labels:
            for sensitive_feature in list_sensitive_features:
                labels_and_sensitive_feature.append((label, sensitive_feature))

        if isinstance(labels, list):
            labels = np.array(labels)
        if isinstance(sensitive_features, list):
            sensitive_features = np.array(sensitive_features)
        to_be_sampled = []
        total_sampled = 0

        # create the distribution for each class
        for label, sensitive_feature in labels_and_sensitive_feature:
            # For each label we want a distribution over the num_partitions
            distribution = np.random.dirichlet(num_partitions * [alpha], size=1)
            # we have to sample from the group of samples that have label equal
            # to label and not from the entire labels list.
            filtered_labels = labels[
                (labels == label) & (sensitive_features == sensitive_feature)
            ]
            tmp_to_be_sampled = np.random.choice(
                num_partitions, len(filtered_labels), p=distribution[0]
            )
            total_sampled += len(tmp_to_be_sampled)
            # Inside to_be_sampled we save a counter for each label
            # The counter is the number of samples that we want to sample for each
            # partition
            to_be_sampled.append(Counter(tmp_to_be_sampled))
        assert total_sampled == len(labels)
        # create the partitions
        partitions_index = {
            f"node_{cluster_name}": [] for cluster_name in range(0, num_partitions)
        }
        total_samples = 0
        for (class_index, _), distribution_samples in zip(
            labels_and_sensitive_feature, to_be_sampled
        ):
            for cluster_name, samples in distribution_samples.items():
                partitions_index[f"node_{cluster_name}"] += index_per_label[
                    class_index
                ][:samples]
                total_samples += samples

                index_per_label[class_index] = index_per_label[class_index][samples:]

        assert total_samples == len(labels)
        total = 0
        for cluster, samples in partitions_index.items():
            total += len(samples)

        assert total == len(labels)

        partitions_labels = {
            cluster: [item.item() for item in labels[samples]]
            for cluster, samples in partitions_index.items()
        }

        return partitions_index, partitions_labels

    def do_partitioning_with_dataset_list(
        num_partitions: int,
        total_num_classes: int,
        labels: list,
        sensitive_features: list,
        alpha=1000000,
    ) -> dict:
        dir_distributions = []
        if not alpha:
            raise ValueError("Alpha must be a positive number")
        idx = torch.tensor(list(range(len(labels))))
        if isinstance(sensitive_features, torch.Tensor):
            sensitive_features = sensitive_features.numpy()

        index_per_label = {}
        for index, label in zip(idx, labels):
            if isinstance(label, torch.Tensor):
                label = label.item()
            if label not in index_per_label:
                index_per_label[label] = []
            index_per_label[label].append(index)

        # in list labels we have the labels of this dataset
        list_labels = {
            item.item() if isinstance(item, torch.Tensor) else item for item in labels
        }
        list_sensitive_features = {
            item.item() if isinstance(item, torch.Tensor) else item
            for item in sensitive_features
        }
        labels_and_sensitive_feature = []
        for label in list_labels:
            for sensitive_feature in list_sensitive_features:
                labels_and_sensitive_feature.append((label, sensitive_feature))

        if isinstance(labels, list):
            labels = np.array(labels)
        if isinstance(sensitive_features, list):
            sensitive_features = np.array(sensitive_features)
        to_be_sampled = []
        total_sampled = 0

        # create the distribution for each class
        for label, sensitive_feature in labels_and_sensitive_feature:
            # For each label we want a distribution over the num_partitions
            distribution = np.random.dirichlet(num_partitions * [alpha], size=1)
            dir_distributions.append((label, sensitive_feature, distribution))
            # we have to sample from the group of samples that have label equal
            # to label and not from the entire labels list.
            filtered_labels = labels[
                (labels == label) & (sensitive_features == sensitive_feature)
            ]
            tmp_to_be_sampled = np.random.choice(
                num_partitions, len(filtered_labels), p=distribution[0]
            )
            total_sampled += len(tmp_to_be_sampled)
            # Inside to_be_sampled we save a counter for each label
            # The counter is the number of samples that we want to sample for each
            # partition
            to_be_sampled.append(Counter(tmp_to_be_sampled))

        assert total_sampled == len(labels)
        # create the partitions
        partitions_index = {
            f"node_{cluster_name}": [] for cluster_name in range(0, num_partitions)
        }
        total_samples = 0
        for (class_index, _), distribution_samples in zip(
            labels_and_sensitive_feature, to_be_sampled
        ):
            for cluster_name, samples in distribution_samples.items():
                partitions_index[f"node_{cluster_name}"] += index_per_label[
                    class_index
                ][:samples]
                total_samples += samples

                index_per_label[class_index] = index_per_label[class_index][samples:]

        assert total_samples == len(labels)
        total = 0
        for cluster, samples in partitions_index.items():
            total += len(samples)

        assert total == len(labels)

        partitions_labels = {
            cluster: [item.item() for item in labels[samples]]
            for cluster, samples in partitions_index.items()
        }

        partitions_index_list = []
        for node_name, indexes in partitions_index.items():
            partitions_index_list.append(indexes)
            node_labels = labels[indexes]
            node_sensitive_features = sensitive_features[indexes]
            labels_and_sensitive_feature = zip(node_labels, node_sensitive_features)
            print(f"Node {node_name} has: ", Counter(labels_and_sensitive_feature))

        return (
            partitions_index,
            partitions_labels,
            partitions_index_list,
            dir_distributions,
        )
