import math
import random
from collections import Counter

import numpy as np
import torch


class MajorityMinorityPartition:
    """This class implements the majority-minority partitioning strategy.

    We want to split the dataset among the nodes based on the target class
    to create unbalanced datasets. Given a dataset with C classes,
    we split the samples of that class into two parts.
    The first one comprising 70% of the data of that class is
    called the majority class. The second one, comprising the remaining 30% is
    called the minority class. To assign majority and minority classes to the clusters
    we have two different cases.
    If n_labels > n_clusters, we know that each node will have
    max(num_labels / num_clusters, 1) different majority labels.
    Considering that n_labels > n_clusters, each label will be assigned at most to
    one node. Sometimes, we will have some labels that are not assigned to any node.
    In this case, we distribute these labels among the nodes with an IID strategy.
    The remaining 30% of the data will be assigned using a different strategy. Each
    of these minority classes will be assigned to 50% of the nodes that do not have
    that class. For instance, let us consider the case with 10 labels and 5 clusters.
    In this case, each cluster will have 2 majority classes. We assign 70% of the data
    of each majority class to one node. Then we have to assign the remaining 30%
    of the data. In this case, each minority class will be assigned to two nodes.
    If n_labels < n_clusters, each majority class will be assigned to at most
    n_clusters / n_labels nodes. In this case, we have that a majority class can be
    assigned to more than one node. In this case, we equally divide the majority
    class's data among the nodes. For the minority classes, we have that each minority
    class will be assigned to the 50% of the nodes that do not have that class.
    For instance. If we have 5 labels and 10 nodes, then we have that each node will
    have 2 majority classes. We assign 35% of the data of each majority class to one
    of these two nodes. Then we have to assign the remaining 30% of the data. In this
    case, each minority class will be assigned to two nodes.
    """

    def do_partitioning(
        dataset: torch.utils.data.Dataset,
        num_partitions: int,
        percentage_majority_class: float = 0.7,
        num_nodes_split_minority=2,  # 2 if we want the 50% of the nodes
    ) -> dict:
        random.seed(43)
        percentage_minority_class = 1 - percentage_majority_class
        labels = dataset.targets
        data = dataset.data if hasattr(dataset, "data") else dataset.samples

        labels_list = sorted(
            {item.item() if isinstance(item, torch.Tensor) else item for item in labels}
        )
        num_labels = len(labels_list)
        partition_names = [f"cluster_{i}" for i in range(num_partitions)]
        counter_partitions = {partition_name: 0 for partition_name in partition_names}

        if num_labels >= num_partitions:
            # In this case, since we have more labels than partitions,
            # each majority class will be assigned only to one partition
            # Each partition will have majority_classes_per_partition majority classes
            majority_classes_per_partition = int(max(num_labels / num_partitions, 1))

            percentage_majority_class_splitted = percentage_majority_class

            # Each minority class will be partitioned into the 50% of the nodes
            # that do not have that class
            n_partitions_minority_classes = int(
                (num_partitions - 1) / num_nodes_split_minority
            )

            # We divide the percentage of the minority class by the number of
            # nodes to which we want to assign the minority class
            percentage_minority_per_partition = round(
                percentage_minority_class / n_partitions_minority_classes,
                2,
            )
            # How many minority classes will be assigned to each partition?
            # In minority_classes_per_partition we store this number
            # Considering that we have majority_classes_per_partition and that we have
            # num_partitions, we will have to assigna at least
            # (num_partitions * majority_classes_per_partition) minority classes.
            # Since we want to assign the minority class to the 50% of the nodes
            # we have to multiply (num_partitions * majority_classes_per_partition) to
            # n_partitions_minority_classes to find the total number of minority classes
            # that we have to assign to the partitions.
            # To find the number of minority classes per partition we divide the total
            # number of minority classes by the number of partitions
            minority_classes_per_partition = int(
                max(
                    (
                        (num_partitions * majority_classes_per_partition)
                        * n_partitions_minority_classes
                    )
                    / num_partitions,
                    1,
                )
            )
            majority_classes = {
                item: percentage_majority_class
                for item in labels_list[
                    : majority_classes_per_partition * num_partitions
                ]
            }

        else:
            majority_classes = {item: percentage_majority_class for item in labels_list}

            # Number of majority classes per partition
            majority_classes_per_partition = int(
                math.ceil(max(num_partitions / num_labels, 1)),
            )

            # Each of the minority classes will be assigned to 50% of the nodes
            # In this case we have a variable number of majority_classes_per_partition
            # so we have to compute this number using this formula.
            n_partitions_minority_classes = int(
                (num_partitions - majority_classes_per_partition)
                / num_nodes_split_minority
            )
            percentage_majority_class_splitted = (
                percentage_majority_class / majority_classes_per_partition
            )
            percentage_minority_per_partition = round(
                (1 - percentage_majority_class) / n_partitions_minority_classes, 2
            )

        (
            percentage_majority_assigned_to_each_partition,
            majority_classes,
        ) = MajorityMinorityPartition.assign_majority_classes(
            partition_names,
            num_labels,
            num_partitions,
            labels_list,
            majority_classes=majority_classes,
            majority_classes_per_partition=majority_classes_per_partition,
            percentage_majority_per_partition=percentage_majority_class_splitted,
        )
        # Now we compute the minority classes that we have to assign
        # to the clusters/nodes
        minority_classes_to_be_assigned = (
            MajorityMinorityPartition.get_minority_classes_to_be_assigned(
                majority_classes=majority_classes,
                n_partitions_minority_classes=n_partitions_minority_classes,
                percentage_minority_per_partition=percentage_minority_per_partition,
            )
        )
        percentage_minority_assigned_to_each_partition = MajorityMinorityPartition.assign_minority_classes(
            minority_classes_to_be_assigned=minority_classes_to_be_assigned,
            counter_partitions=counter_partitions,
            n_partitions_minority_classes=minority_classes_per_partition
            if num_labels >= num_partitions
            else n_partitions_minority_classes,
            percentage_majority_assigned_to_each_partition=percentage_majority_assigned_to_each_partition,
            partition_names=partition_names,
        )
        final_percentage_assignment = MajorityMinorityPartition.merge_dictionaries(
            percentage_minority_assigned_to_each_partition=percentage_minority_assigned_to_each_partition,
            percentage_majority_assigned_to_each_partition=percentage_majority_assigned_to_each_partition,
            labels_list=labels_list,
            num_partitions=num_partitions,
        )

        # assert that the sum of the percentages is 1
        percentages = {item: 0 for item in labels_list}
        for partition_distribution in final_percentage_assignment.values():
            for class_, percentage in partition_distribution.items():
                percentages[class_] += percentage

        for item, percentage in percentages.items():
            assert torch.isclose(torch.tensor(percentage), torch.tensor(1.0), atol=1e-1)

        # Now that we have the percentages for each partition, we want to assign
        # the indices to the partitions

        (
            splitted_labels,
            splitted_data,
            splitted_indexes,
        ) = MajorityMinorityPartition.split_labels_and_indices(
            dataset=dataset,
            labels_list=labels_list,
            final_percentage_assignment=final_percentage_assignment,
            data=data,
        )

        return splitted_indexes, splitted_labels, splitted_data

    def assign_majority_classes(
        partition_names,
        num_labels,
        num_partitions,
        labels_list,
        majority_classes,
        majority_classes_per_partition,
        percentage_majority_per_partition,
    ):
        percentage_majority_assigned_to_each_partition = {
            partition_id: {} for partition_id in partition_names
        }
        assigned_majority_classes = set()
        counter_partitions = {partition_name: 0 for partition_name in partition_names}
        sorted_values = [
            item[0] for item in sorted(counter_partitions.items(), key=lambda x: x[1])
        ]
        added = False

        while majority_classes:
            # for partition_id in percentage_majority_assigned_to_each_partition:
            for partition_id in sorted_values:
                if majority_classes:
                    # random sample one of the majority classes
                    majority_class = random.choice(list(majority_classes.keys()))

                    majority_classes[
                        majority_class
                    ] -= percentage_majority_per_partition
                    if majority_classes[majority_class] <= 0:
                        majority_classes.pop(majority_class)

                    percentage_majority_assigned_to_each_partition[partition_id][
                        majority_class
                    ] = percentage_majority_per_partition
                    assigned_majority_classes.add(majority_class)
                    counter_partitions[partition_id] += 1

        return percentage_majority_assigned_to_each_partition, assigned_majority_classes

    def assign_minority_classes(
        minority_classes_to_be_assigned,
        counter_partitions,
        n_partitions_minority_classes,
        percentage_majority_assigned_to_each_partition,
        partition_names,
    ):
        # We want to assign the minority classes to the partitions
        percentage_minority_assigned_to_each_partition = {
            partition_id: {} for partition_id in partition_names
        }
        while minority_classes_to_be_assigned:
            minority_class = minority_classes_to_be_assigned.pop()
            minority_class_name = minority_class[0]
            minority_class_percentage = minority_class[1]
            sorted_values = [
                item[0]
                for item in sorted(counter_partitions.items(), key=lambda x: x[1])
            ]
            added = False
            for partition_id in sorted_values:
                if (
                    len(
                        percentage_minority_assigned_to_each_partition.get(
                            partition_id, []
                        ),
                    )
                    < n_partitions_minority_classes
                    and minority_class_name
                    not in percentage_majority_assigned_to_each_partition[
                        partition_id
                    ].keys()
                    and minority_class_name
                    not in percentage_minority_assigned_to_each_partition[
                        partition_id
                    ].keys()
                ):
                    percentage_minority_assigned_to_each_partition[partition_id][
                        minority_class_name
                    ] = minority_class_percentage

                    counter_partitions[partition_id] += 1
                    added = True
                    break
            if not added:
                minority_classes_to_be_assigned.append(minority_class)
        return percentage_minority_assigned_to_each_partition

    def get_minority_classes_to_be_assigned(
        majority_classes,
        n_partitions_minority_classes,
        percentage_minority_per_partition,
    ):
        majority_classes = list(majority_classes)
        minority_classes_to_be_assigned = []
        for item in majority_classes:
            for _ in range(n_partitions_minority_classes):
                minority_classes_to_be_assigned.append(
                    (item, percentage_minority_per_partition)
                )
        return minority_classes_to_be_assigned

    def merge_dictionaries(
        percentage_minority_assigned_to_each_partition,
        percentage_majority_assigned_to_each_partition,
        labels_list,
        num_partitions,
    ):
        final_percentage_assignment = {}

        for partition_name in percentage_minority_assigned_to_each_partition:
            final_percentage_assignment[partition_name] = {
                **percentage_minority_assigned_to_each_partition[partition_name],
                **percentage_majority_assigned_to_each_partition[partition_name],
            }
        # Now we want to check if we have unassigned labels
        assigned_labels = set()
        for assignments in final_percentage_assignment.values():
            assigned_labels |= set(list(assignments.keys()))

        unassigned_labels = list(set(labels_list) - assigned_labels)
        # If we still have unassigned labels, we want to assign them to the
        # partitions with a IID strategy
        if len(unassigned_labels) > 0:
            for partition_name in final_percentage_assignment:
                for unassigned_label in unassigned_labels:
                    final_percentage_assignment[partition_name][unassigned_label] = (
                        1 / num_partitions
                    )

        return final_percentage_assignment

    def split_labels_and_indices(
        dataset,
        labels_list,
        final_percentage_assignment,
        data,
    ):
        labels = [
            item.item() if isinstance(item, torch.Tensor) else item
            for item in dataset.targets
        ]
        counter_indices = Counter(labels)
        indices_per_class = {label: [] for label in labels_list}
        for index, label in enumerate(labels):
            indices_per_class[label].append(index)

        for label in labels_list:
            random.shuffle(indices_per_class[label])
        splitted_indexes = {}
        for partition in final_percentage_assignment:
            tmp_ = []
            for label in final_percentage_assignment[partition]:
                percentage = final_percentage_assignment[partition][label]
                num_samples = int(percentage * counter_indices[label])
                tmp_ += indices_per_class[label][:num_samples]
                indices_per_class[label] = indices_per_class[label][num_samples:]
            splitted_indexes[partition] = tmp_
        total_indexes = 0
        for item in splitted_indexes:
            total_indexes += len(item)
        splitted_labels = {
            partition_name: [
                item.item() if isinstance(item, torch.Tensor) else item
                for item in dataset.targets[indexes]
            ]
            for partition_name, indexes in splitted_indexes.items()
        }

        splitted_data = {
            partition_name: np.array(data)[indexes]
            for partition_name, indexes in splitted_indexes.items()
        }
        return splitted_labels, splitted_data, splitted_indexes
