import os
import shutil
from collections import Counter

import numpy as np
import torch

from fl_puf.FederatedDataset.PartitionTypes.iid_partition import IIDPartition
from fl_puf.FederatedDataset.PartitionTypes.majority_minority_partition import (
    MajorityMinorityPartition,
)
from fl_puf.FederatedDataset.PartitionTypes.non_iid_partition import NonIIDPartition
from fl_puf.FederatedDataset.PartitionTypes.non_iid_partition_nodes import (
    NonIIDPartitionNodes,
)
from fl_puf.FederatedDataset.PartitionTypes.non_iid_partition_nodes_with_sensitive_feature import (
    NonIIDPartitionNodesWithSensitiveFeature,
)
from fl_puf.FederatedDataset.PartitionTypes.non_iid_partition_with_sensitive_feature import (
    NonIIDPartitionWithSensitiveFeature,
)
from fl_puf.FederatedDataset.Utils.custom_dataset import MyDataset, MyDatasetWithCSV
from fl_puf.FederatedDataset.Utils.dataset_downloader import DatasetDownloader
from fl_puf.FederatedDataset.Utils.preferences import Preferences


class FederatedDataset:
    def generate_partitioned_dataset(
        config: Preferences = None,
        split_type_clusters: str = None,
        split_type_nodes: str = None,
        num_nodes: int = None,
        num_clusters: int = None,
        num_classes: int = None,
        alpha: float = None,
        dataset_name: str = None,
        custom_dataset: dict = None,
        store_path: str = None,
        train_ds: torch.utils.data.Dataset = None,
        test_ds: torch.utils.data.Dataset = None,
    ) -> None:
        split_type_clusters = (
            config.data_split_config.split_type_clusters
            if config
            else split_type_clusters
        )
        split_type_nodes = (
            config.data_split_config.split_type_nodes if config else split_type_nodes
        )
        num_nodes = config.data_split_config.num_nodes if config else num_nodes
        num_clusters = config.data_split_config.num_clusters if config else num_clusters
        num_classes = config.data_split_config.num_classes if config else num_classes
        alpha = config.data_split_config.alpha if config else alpha
        dataset_name = config.dataset if config else dataset_name
        store_path = config.data_split_config.store_path if config else store_path

        if train_ds is None and test_ds is None:
            train_ds, test_ds = DatasetDownloader.download_dataset(
                dataset_name=dataset_name,
            )
        data = train_ds.data if hasattr(train_ds, "data") else train_ds.samples
        cluster_splits_train = []
        cluster_splits_test = []

        # At the moment we only consider two cases. The first one is the case in which
        # we want to split the dataset both among the clusters and among the nodes.
        # This is used for the experiments for the Federated learning with P2P.
        # In particular, at the moment the only split type supported for this case is
        # the majority minority in the cluster and the non iid among the nodes.
        # Other combinations of splits could work but I'm not sure about the results.
        # The other case is the one in which we only split the dataset among
        # the nodes. This is a classic Federated learning scenario.
        if num_nodes and num_clusters:
            # First we split the data among the clusters
            (
                splitted_indexes_train,
                labels_per_cluster_train,
                samples_per_cluster_train,
            ) = FederatedDataset.partition_data(
                data=train_ds,
                split_type=split_type_clusters,
                num_partitions=num_clusters,
                alpha=alpha,
                num_classes=num_classes,
            )
            (
                splitted_indexes_test,
                labels_per_cluster_test,
                samples_per_cluster_test,
            ) = FederatedDataset.partition_data(
                data=test_ds,
                split_type=split_type_clusters,
                num_partitions=num_clusters,
                alpha=alpha,
                num_classes=num_classes,
            )
            # And then we split the data among the nodes
            for cluster_name in splitted_indexes_train:
                current_labels_train = labels_per_cluster_train[cluster_name]
                current_samples_train = samples_per_cluster_train[cluster_name]

                current_labels_test = labels_per_cluster_test[cluster_name]
                current_samples_test = samples_per_cluster_test[cluster_name]

                (
                    splitted_indexes_train_nodes,
                    labels_per_cluster_train_nodes,
                    samples_per_cluster_train_nodes,
                ) = FederatedDataset.partition_data(
                    data=MyDataset(
                        samples=current_samples_train,
                        targets=current_labels_train,
                        transform=train_ds.transform,
                    ),
                    split_type=split_type_nodes,
                    num_partitions=num_nodes,
                    alpha=alpha,
                    num_classes=num_classes,
                )
                cluster_splits_train.append(
                    (
                        cluster_name,
                        splitted_indexes_train_nodes,
                        labels_per_cluster_train_nodes,
                    )
                )
                (
                    splitted_indexes_test_nodes,
                    labels_per_cluster_test_nodes,
                    samples_per_cluster_test_nodes,
                ) = FederatedDataset.partition_data(
                    data=MyDataset(
                        samples=current_samples_test,
                        targets=current_labels_test,
                        transform=train_ds.transform,
                    ),
                    split_type=split_type_nodes,
                    num_partitions=num_nodes,
                    alpha=alpha,
                    num_classes=num_classes,
                )

                cluster_splits_test.append(
                    (
                        cluster_name,
                        splitted_indexes_test_nodes,
                        labels_per_cluster_test_nodes,
                    )
                )
                # At the end of this loop we have a list of tuples. Each tuple contains
                # The name of the clusters, the indexes of the samples for each node
                # and the labels for each node.

        elif num_nodes:
            # In this second case we only want to partition the dataset among
            # the nodes.
            (
                splitted_indexes_train,
                labels_per_cluster_train,
                samples_per_cluster_train,
            ) = FederatedDataset.partition_data(
                data=train_ds,
                split_type=split_type_nodes,
                num_partitions=num_nodes,
                alpha=alpha,
                num_classes=num_classes,
            )
            (
                splitted_indexes_test,
                labels_per_cluster_test,
                samples_per_cluster_test,
            ) = FederatedDataset.partition_data(
                data=test_ds,
                split_type=split_type_nodes,
                num_partitions=num_nodes,
                alpha=alpha,
                num_classes=num_classes,
            )

        # We take some information from the dataset. We need these information
        # because then we will create new partitioned datasets that will
        # contain these informations.
        dataset_with_csv = False
        if hasattr(train_ds, "image_path"):
            image_path_train = train_ds.image_path
            samples_train = train_ds.samples
            image_path_test = test_ds.image_path
            samples_test = test_ds.samples
            dataset_with_csv = True
        else:
            image_path_train = None
            image_path_test = None

        if hasattr(train_ds, "data"):
            samples_train = train_ds.data
            samples_test = test_ds.data
        transform_train = train_ds.transform
        transform_test = train_ds.transform

        targets_train_per_class = FederatedDataset.create_targets_per_class(
            data=train_ds,
        )
        targets_test_per_class = FederatedDataset.create_targets_per_class(
            data=test_ds,
        )

        partitions_train = None
        partitions_test = None
        # Finally, we create the splitted datasets
        # There are two cases: the one in which we splitted both among the
        # clusters and among the nodes and the one in which we splitted only
        # among the nodes
        if cluster_splits_train:
            partitions_train = (
                FederatedDataset.create_partitioned_dataset_with_clusters(
                    cluster_splits=cluster_splits_train,
                    targets_per_class=targets_train_per_class,
                    dataset_with_csv=dataset_with_csv,
                    dataset=train_ds,
                    image_path=image_path_train,
                    transform=transform_train,
                )
            )

            partitions_test = FederatedDataset.create_partitioned_dataset_with_clusters(
                cluster_splits=cluster_splits_test,
                targets_per_class=targets_test_per_class,
                dataset_with_csv=dataset_with_csv,
                dataset=test_ds,
                image_path=image_path_test,
                transform=transform_test,
            )

        else:
            partitions_train = FederatedDataset.create_partitioned_dataset(
                labels_per_cluster=labels_per_cluster_train,
                targets_per_class=targets_train_per_class,
                dataset_with_csv=dataset_with_csv,
                dataset=train_ds,
                image_path=image_path_train,
                transform=transform_train,
            )
            partitions_test = FederatedDataset.create_partitioned_dataset(
                labels_per_cluster=labels_per_cluster_test,
                targets_per_class=targets_test_per_class,
                dataset_with_csv=dataset_with_csv,
                dataset=test_ds,
                image_path=image_path_test,
                transform=transform_train,
            )

        if os.path.exists(store_path):
            shutil.rmtree(store_path)
        os.makedirs(store_path)

        # Now we can store the partitioned datasets
        FederatedDataset.store_partitioned_datasets(
            partitions_train,
            store_path=store_path,
            split_name="train",
        )
        FederatedDataset.store_partitioned_datasets(
            partitions_test,
            store_path=store_path,
            split_name="test",
        )

    def create_targets_per_class(data):
        """This function creates a dictionary containing the targets of
        the datasets as key and the indexes of the samples belonging to
        that class as values.

        Args:
            data (_type_): the dataset we are partitioning
        """
        targets_per_class = {}

        for index, target in enumerate(data.targets):
            target = target.item() if isinstance(target, torch.Tensor) else target
            if target not in targets_per_class:
                targets_per_class[target] = []
            targets_per_class[target].append(
                index.item() if isinstance(index, torch.Tensor) else index,
            )
        return targets_per_class

    def store_partitioned_datasets(
        partitioned_datasets: list,
        store_path: str,
        split_name: str,
    ):
        for index, partition_name in enumerate(partitioned_datasets):
            if isinstance(partitioned_datasets[partition_name], dict):
                for node_name in partitioned_datasets[partition_name]:
                    torch.save(
                        partitioned_datasets[partition_name][node_name],
                        f"{store_path}/{partition_name}_{node_name}_{split_name}.pt",
                    )
            else:
                # When we split the dataset only among the nodes we want to
                # store the partitioned dataset in folders. Each folder will
                # contain the partitioned dataset of a node. This is
                # useful when we want to use Flower to train the federated model.
                directory = f"{store_path}/{index}/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                torch.save(
                    partitioned_datasets[partition_name],
                    f"{directory}{split_name}.pt",
                )

    def partition_data(
        data,
        split_type: str,
        num_partitions: int,
        alpha: float,
        num_classes: int,
    ):
        samples_per_cluster_train = []
        if split_type == "iid":
            # Partition the training dataset among the clusters
            (
                splitted_indexes_train,
                labels_per_cluster_train,
            ) = IIDPartition.do_partitioning(
                dataset=data,
                num_partitions=num_partitions,
                total_num_classes=num_classes,
            )

        elif split_type == "majority_minority":
            (
                splitted_indexes_train,
                labels_per_cluster_train,
                samples_per_cluster_train,
            ) = MajorityMinorityPartition.do_partitioning(
                dataset=data,
                num_partitions=num_partitions,
            )

        elif split_type == "non_iid":
            (
                splitted_indexes_train,
                labels_per_cluster_train,
            ) = NonIIDPartition.do_partitioning(
                dataset=data,
                num_partitions=num_partitions,
                total_num_classes=num_classes,
                alpha=alpha,
            )
        elif split_type == "non_iid_nodes":
            (
                splitted_indexes_train,
                labels_per_cluster_train,
            ) = NonIIDPartitionNodes.do_partitioning(
                dataset=data,
                num_partitions=num_partitions,
                total_num_classes=num_classes,
                alpha=alpha,
            )
        elif split_type == "non_iid_sensitive_feature":
            (
                splitted_indexes_train,
                labels_per_cluster_train,
            ) = NonIIDPartitionWithSensitiveFeature.do_partitioning(
                dataset=data,
                num_partitions=num_partitions,
                total_num_classes=num_classes,
                alpha=alpha,
            )
        elif split_type == "non_iid_nodes_sensitive_feature":
            (
                splitted_indexes_train,
                labels_per_cluster_train,
            ) = NonIIDPartitionNodesWithSensitiveFeature.do_partitioning(
                dataset=data,
                num_partitions=num_partitions,
                total_num_classes=num_classes,
                alpha=alpha,
            )
        return (
            splitted_indexes_train,
            labels_per_cluster_train,
            samples_per_cluster_train,
        )

    def create_partitioned_dataset(
        labels_per_cluster,
        targets_per_class,
        dataset_with_csv,
        dataset,
        image_path,
        transform,
    ):
        partitions = {}
        for partition_name, labels in labels_per_cluster.items():
            counter_labels = Counter(labels)
            indexes = []
            for label, count in counter_labels.items():
                if isinstance(label, torch.Tensor):
                    label = label.item()
                indexes += targets_per_class[label][:count]
                targets_per_class[label] = targets_per_class[label][count:]
            if dataset_with_csv:
                if isinstance(dataset.targets, list):
                    dataset.targets = torch.tensor(dataset.targets)
                train_partition = MyDatasetWithCSV(
                    targets=dataset.targets[indexes],
                    image_path=image_path,
                    image_ids=np.array(dataset.samples)[indexes],
                    transform=transform,
                    sensitive_features=np.array(dataset.sensitive_features)[indexes]
                    if hasattr(dataset, "sensitive_features")
                    else None,
                )
            else:
                train_partition = MyDataset(
                    targets=dataset.targets[indexes],
                    samples=torch.tensor(dataset.data)[indexes].to(torch.float32),
                    transform=transform,
                )
            partitions[partition_name] = train_partition
        for partition_name, partition in partitions.items():
            if hasattr(partition, "sensitive_features"):
                print(
                    f"Partition {partition_name} has {len(partition)} samples, {len(set([item.item() for item in partition.sensitive_features]))} sensitive_features: {Counter([(target.item(), feature.item()) for target, feature in zip(partition.targets, partition.sensitive_features)])}",
                )
            else:
                print(
                    f"Partition {partition_name} has {len(partition)} samples: {Counter([item.item() for item in partition.targets])}",
                )
        print("_________________________")

        return partitions

    def create_partitioned_dataset_with_clusters(
        cluster_splits,
        targets_per_class,
        dataset_with_csv,
        dataset,
        image_path,
        transform,
    ):
        partitions = {}
        counter_partitions = {}
        for (
            cluster_name,
            _,
            splitted_labels_cluster,
        ) in cluster_splits:
            counter_partitions[cluster_name] = []
            partitions[cluster_name] = {}

            # Now we have the indexes of the data for each node
            # We can create the dataset for each node
            for node_name, labels in splitted_labels_cluster.items():
                counter_labels = Counter(labels)
                indexes = []
                for label, count in counter_labels.items():
                    indexes += targets_per_class[label][:count]
                    targets_per_class[label] = targets_per_class[label][count:]
                if dataset_with_csv:
                    train_partition = MyDatasetWithCSV(
                        targets=dataset.targets[indexes],
                        image_path=image_path,
                        image_ids=torch.tensor(dataset.samples)[indexes],
                        transform=transform,
                        sensitive_features=torch.tensor(dataset.sensitive_features)[
                            indexes
                        ]
                        if hasattr(dataset, "sensitive_features")
                        else None,
                    )
                    partitions[cluster_name][node_name] = train_partition
                else:
                    train_partition = MyDataset(
                        targets=dataset.targets[indexes],
                        samples=np.array(dataset.data)[indexes],
                        transform=transform,
                    )
                    partitions[cluster_name][node_name] = train_partition
                counter_partitions[cluster_name].append(counter_labels)

        for cluster_name in counter_partitions:
            print(
                f"Cluster {cluster_name} has {sum(counter_partitions[cluster_name], Counter())}",
            )
        print("_________________________")
        return partitions
