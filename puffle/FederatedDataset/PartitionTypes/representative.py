import itertools
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import torch

from DPL.Regularization.RegularizationLoss import RegularizationLoss


class Representative:
    def do_partitioning(
        labels: np.ndarray,
        sensitive_features: np.ndarray,
        num_partitions: int,
        total_num_classes: int,
        group_to_reduce: tuple,
        group_to_increment: tuple,
        ratio_unfair_nodes: float,
        ratio_unfairness: float,
        number_of_samples_per_node: int = None,
        one_group_nodes: bool = False,
    ) -> list:
        sensitive_features = np.array([item.item() for item in sensitive_features])
        labels = np.array([item for item in labels])
        labels_and_sensitive = list(zip(labels, sensitive_features))
        indexes = list(range(len(labels)))

        samples_per_node = (
            number_of_samples_per_node
            if number_of_samples_per_node is not None
            else len(labels) // num_partitions
        )
        np.random.shuffle(indexes)

        # Distribute the data among the nodes with a random sample from the dataset
        # considering the number of samples per node
        nodes = []
        for i in range(num_partitions):
            nodes.append([])
            nodes[i].extend(indexes[:samples_per_node])
            indexes = indexes[samples_per_node:]

        # Create the dictionary with the remaining data
        remaining_data = {}
        for index in indexes:
            if labels_and_sensitive[index] not in remaining_data:
                remaining_data[labels_and_sensitive[index]] = []
            remaining_data[labels_and_sensitive[index]].append(index)

        number_unfair_nodes = (
            int(num_partitions * ratio_unfair_nodes)
            if ratio_unfair_nodes is not None
            else 0
        )
        number_fair_nodes = num_partitions - number_unfair_nodes

        # At the moment this is the only thing that is working, we need
        # to fix the opposite direction version
        fair_nodes, unfair_nodes = Representative.create_unfair_nodes(
            fair_nodes=nodes[:number_fair_nodes],
            nodes_to_unfair=nodes[number_fair_nodes:],
            remaining_data=remaining_data,
            group_to_reduce=group_to_reduce,
            group_to_increment=group_to_increment,
            ratio_unfairness=ratio_unfairness,
            combination=labels_and_sensitive,
        )
        # modify the unfair nodes so that they only have samples
        # with sensitive value equal to 1 in half of the unfair nodes

        if one_group_nodes:
            new_unfair_nodes = []
            for node_id, indexes in enumerate(unfair_nodes):
                sensitive_features_list = sensitive_features[indexes]
                labels_list = labels[indexes]
                combinations = list(zip(labels_list, sensitive_features_list))
                tmp_indexes = []
                for sf, lb, index in zip(sensitive_features_list, labels_list, indexes):
                    if (sf, lb) != (1, 1):
                        tmp_indexes.append(index)

                new_unfair_nodes.append(tmp_indexes)
            unfair_nodes = new_unfair_nodes

        predictions = [labels[indexes] for indexes in fair_nodes + unfair_nodes]
        sensitive_features = [
            sensitive_features[indexes] for indexes in fair_nodes + unfair_nodes
        ]

        disparities = Representative.compute_disparities_debug(
            predictions=predictions, sensitive_features=sensitive_features
        )
        counter_distribution_nodes = Representative.compute_distribution_debug(
            predictions=predictions, sensitive_features=sensitive_features
        )
        # Representative.plot_distributions(
        #     title="Distribution of the nodes",
        #     counter_groups=counter_distribution_nodes,
        #     nodes=[f"{i}" for i in range(len(nodes))],
        # )
        # print(disparities)
        # Representative.plot_bar_plot(
        #     title="Disparities",
        #     disparities=disparities,
        #     nodes=[f"{i}" for i in range(len(nodes))],
        # )
        # size_of_each_client_data = [len(client) for client in fair_nodes + unfair_nodes]
        # Representative.plot_bar_plot(
        #     title="Client Size",
        #     disparities=size_of_each_client_data,
        #     nodes=[f"{i}" for i in range(len(nodes))],
        # )
        return (
            fair_nodes + unfair_nodes,
            [0] * len(fair_nodes) + [1] * len(unfair_nodes),
        )

    def compute_disparities_debug(predictions, sensitive_features):
        disparities = []
        for prediction, sensitive_feature in zip(predictions, sensitive_features):
            max_disparity = np.max(
                [
                    RegularizationLoss().compute_violation_with_argmax(
                        predictions_argmax=prediction,
                        sensitive_attribute_list=sensitive_feature,
                        current_target=target,
                        current_sensitive_feature=sv,
                    )
                    for target in range(0, 1)
                    for sv in range(0, 1)
                ]
            )
            disparities.append(max_disparity)
        print(f"Mean of disparity {np.mean(disparities)} - std {np.std(disparities)}")
        return disparities

    def compute_distribution_debug(predictions, sensitive_features):
        counter_nodes = []
        for prediction, sensitive_feature in zip(predictions, sensitive_features):
            counter_node = []
            for pred, sf in zip(prediction, sensitive_feature):
                counter_node.append((pred, sf))
            counter_nodes.append(Counter(counter_node))
        return counter_nodes

    def plot_distributions(title: str, counter_groups: list, nodes: list):
        counter_group_0_0 = [counter[(0, 0)] for counter in counter_groups]
        counter_group_0_1 = [counter[(0, 1)] for counter in counter_groups]
        counter_group_1_0 = [counter[(1, 0)] for counter in counter_groups]
        counter_group_1_1 = [counter[(1, 1)] for counter in counter_groups]

        # plot a barplot with counter_group_0_0, counter_group_0_1, counter_group_1_0, counter_group_1_1
        # for each client in the same plot
        plt.figure(figsize=(20, 8))

        plt.bar(range(len(counter_group_0_0)), counter_group_0_0)
        plt.bar(
            range(len(counter_group_0_1)), counter_group_0_1, bottom=counter_group_0_0
        )
        plt.bar(
            range(len(counter_group_1_0)),
            counter_group_1_0,
            bottom=[sum(x) for x in zip(counter_group_0_0, counter_group_0_1)],
        )
        plt.bar(
            range(len(counter_group_1_1)),
            counter_group_1_1,
            bottom=[
                sum(x)
                for x in zip(counter_group_0_0, counter_group_0_1, counter_group_1_0)
            ],
        )

        plt.xlabel("Client")
        plt.ylabel("Amount of samples")
        plt.title("Samples for each group (target/sensitive Value) per client")
        plt.legend(["0,0", "0,1", "1,0", "1,1"])
        # font size 20
        plt.rcParams.update({"font.size": 20})
        plt.rcParams.update({"font.size": 10})
        plt.savefig(f"./{title}.png")
        plt.tight_layout()

    # plot the bar plot of the disparities
    def plot_bar_plot(title: str, disparities: list, nodes: list):
        plt.figure(figsize=(20, 8))
        plt.bar(range(len(disparities)), disparities)
        plt.xticks(range(len(nodes)), nodes)
        plt.title(title)
        # add a vertical line on xtick=75
        plt.axvline(x=75, color="r", linestyle="--")
        plt.xticks(rotation=90)
        # plt.show()
        # font size x axis
        plt.rcParams.update({"font.size": 10})
        plt.savefig(f"./{title}.png")
        plt.tight_layout()

    def create_unfair_nodes(
        fair_nodes: list,
        nodes_to_unfair: list,
        remaining_data: dict,
        group_to_reduce: tuple,
        group_to_increment: tuple,
        ratio_unfairness: tuple,
        combination: list,
    ):
        """
        This function creates the unfair nodes. It takes the nodes that we want to be unfair and the remaining data
        and it returns the unfair nodes created by reducing the group_to_reduce and incrementing the group_to_increment
        based on the ratio_unfairness

        params:
        nodes_to_unfair: list of nodes that we want to make unfair
        remaining_data: dictionary with the remaining data that we will use to replace the
            samples that we remove from the nodes_to_unfair
        group_to_reduce: the group that we want to be unfair. For instance, in the case of binary target and binary sensitive value
            we could have (0,0), (0,1), (1,0) or (1,1)
        group_to_increment: the group that we want to increment. For instance, in the case of binary target and binary sensitive value
            we could have (0,0), (0,1), (1,0) or (1,1)
        ratio_unfairness: tuple (min, max) where min is the minimum ratio of samples that we want to remove from the group_to_reduce
        """
        # assert (
        #     remaining_data[group_to_reduce] != []
        # ), "Choose a different group to be unfair"
        # remove the samples from the group that we want to be unfair
        unfair_nodes = []
        number_of_samples_to_add = []
        removed_samples = []

        for node in nodes_to_unfair:
            node_data = []
            count_sensitive_group_samples = 0
            # We count how many sample each node has from the group that we want to reduce
            for sample in node:
                if combination[sample] == group_to_reduce:
                    count_sensitive_group_samples += 1

            # We compute the number of samples that we want to remove from the group_to_reduce
            # based on the ratio_unfairness
            current_ratio = np.random.uniform(ratio_unfairness[0], ratio_unfairness[1])
            samples_to_be_removed = int(count_sensitive_group_samples * current_ratio)
            number_of_samples_to_add.append(samples_to_be_removed)

            for sample in node:
                # Now we remove the samples from the group_to_reduce
                # and we store them in removed_samples
                if combination[sample] == group_to_reduce and samples_to_be_removed > 0:
                    samples_to_be_removed -= 1
                    removed_samples.append(sample)
                else:
                    node_data.append(sample)
            unfair_nodes.append(node_data)

        # Now we have to distribute the removed samples among the fair nodes
        max_samples_to_add = len(removed_samples) // len(fair_nodes)
        for node in fair_nodes:
            node.extend(removed_samples[:max_samples_to_add])
            removed_samples = removed_samples[max_samples_to_add:]

        if group_to_increment:
            # Now we have to remove the samples from the group_to_increment
            # from the fair_nodes based on the number_of_samples_to_add
            for node in fair_nodes:
                samples_to_remove = sum(number_of_samples_to_add) // len(fair_nodes)
                for index, sample in enumerate(node):
                    if (
                        combination[sample] == group_to_increment
                        and samples_to_remove > 0
                    ):
                        if combination[sample] not in remaining_data:
                            remaining_data[group_to_increment] = []
                        remaining_data[group_to_increment].append(sample)
                        samples_to_remove -= 1
                        node.pop(index)
            #     if sum(number_of_samples_to_add) > 0:
            #         assert samples_to_remove == 0, "Not enough samples to remove"

            # assert sum(number_of_samples_to_add) <= len(
            #     remaining_data[group_to_increment]
            # ), "Too many samples to add"
            # now we have to add the same amount of data taken from group_to_unfair
            for node, samples_to_add in zip(unfair_nodes, number_of_samples_to_add):
                node.extend(remaining_data[group_to_increment][:samples_to_add])
                remaining_data[group_to_increment] = remaining_data[group_to_increment][
                    samples_to_add:
                ]

        return fair_nodes, unfair_nodes
