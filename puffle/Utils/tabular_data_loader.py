import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from Utils.dutch import TabularDataset
from Utils.utils import Utils
from matplotlib.pyplot import figure
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler

##############################################################################################################


def plot_distribution(distributions, title):
    counters = {}
    key_list = set()
    nodes_data = []
    for distribution in distributions:
        counter = Counter(distribution)
        nodes_data.append(counter)
        for key, value in counter.items():
            key_list.add(key)

    for node_data in nodes_data:
        for key in key_list:
            if key not in node_data:
                if key not in counters:
                    counters[key] = []
                counters[key].append(0)
            else:
                if key not in counters:
                    counters[key] = []
                counters[key].append(node_data[key])

    figure(figsize=(25, 8), dpi=80)
    indexes = np.arange(len(distributions))

    legend_values = []
    colors = ["blue", "green", "red", "purple", "orange", "yellow", "pink", "brown"]
    markers = ["+", "o", "x", "*", "v", "^", "<", ">"]
    for i, (key, value) in enumerate(counters.items()):
        plt.scatter(
            indexes, value, marker=markers[i], linewidths=4, s=100, color=colors[i]
        )
        legend_values.append(key)
    plt.xticks(indexes, list(range(0, len(indexes))))
    plt.rcParams.update({"font.size": 16})
    plt.xlabel("Nodes")
    plt.ylabel("Samples")
    plt.xticks(rotation=90)
    plt.title(title)
    plt.legend(legend_values)
    plt.savefig("distribution.png")


#########################################################################################


def get_tabular_data(
    num_clients: int,
    do_iid_split: bool,
    dataset_name: str,
    num_sensitive_features: int,
    approach: str,
    num_nodes: int,
    ratio_unfair_nodes: float,
    opposite_direction: bool,
    ratio_unfairness: tuple,
    dataset_path=None,
    group_to_reduce: tuple = None,
    group_to_increment: tuple = None,
    number_of_samples_per_node: int = None,
    opposite_group_to_reduce: tuple = None,
    opposite_group_to_increment: tuple = None,
    opposite_ratio_unfairness: tuple = None,
    one_group_nodes: bool = False,
):
    X, z, y = get_tabular_numpy_dataset(
        dataset_name=dataset_name,
        num_sensitive_features=num_sensitive_features,
        dataset_path=dataset_path,
    )
    z = z[:, 0]
    print(f"Data shapes: x={X.shape}, y={y.shape}, z={z.shape}")
    # Prepare training data held by each client
    # Metadata is a list with 0 if the client is fair, 0 otherwise
    client_data, metadata = generate_clients_biased_data_mod(
        X=X,
        y=y,
        z=z,
        approach=approach,
        num_nodes=num_nodes,
        ratio_unfair_nodes=ratio_unfair_nodes,
        opposite_direction=opposite_direction,
        ratio_unfairness=ratio_unfairness,
        group_to_reduce=group_to_reduce,
        group_to_increment=group_to_increment,
        number_of_samples_per_node=number_of_samples_per_node,
        opposite_group_to_reduce=opposite_group_to_reduce,
        opposite_group_to_increment=opposite_group_to_increment,
        opposite_ratio_unfairness=opposite_ratio_unfairness,
        one_group_nodes=one_group_nodes,
    )
    disparities = Utils.compute_disparities_debug(client_data)
    Utils.plot_bar_plot(
        title=f"{approach}",
        disparities=disparities,
        nodes=[f"{i}" for i in range(len(client_data))],
    )
    print(disparities)

    return client_data, disparities, metadata  # , N_is, props_positive


def egalitarian_approach(X, y, z, num_nodes, number_of_samples_per_node=None):
    """
    With this approach we want to distribute the data among the nodes in an egalitarian way.
    This means that each node has the same amount of data and the same ratio of each group

    params:
    X: numpy array of shape (N, D) where N is the number of samples and D is the number of features
    y: numpy array of shape (N, ) where N is the number of samples. Here we have the samples labels
    z: numpy array of shape (N, ) where N is the number of samples. Here we have the samples sensitive features
    num_nodes: number of nodes to generate
    number_of_samples_per_node: number of samples that we want in each node. Can be None, in this case we just use
        len(y)//num_nodes
    """
    combinations = [(target, sensitive_value) for target, sensitive_value in zip(y, z)]
    possible_combinations = set(combinations)
    data = {}
    for combination, x_, y_, z_ in zip(combinations, X, y, z):
        if combination not in data:
            data[combination] = []
        data[combination].append({"x": x_, "y": y_, "z": z_})

    samples_from_each_group = min(list(Counter(combinations).values())) // num_nodes

    if number_of_samples_per_node:
        assert (
            samples_from_each_group * len(possible_combinations)
            >= number_of_samples_per_node
        ), "Too many samples per node, choose a different number of samples per node"
        if (
            samples_from_each_group * len(possible_combinations)
            >= number_of_samples_per_node
        ):
            to_be_removed = (
                samples_from_each_group * len(possible_combinations)
                - number_of_samples_per_node
            ) // len(possible_combinations)
            samples_from_each_group -= to_be_removed

    # create the nodes
    nodes = []
    for i in range(num_nodes):
        nodes.append([])
        # fill the nodes
        for combination in data:
            nodes[i].extend(data[combination][:samples_from_each_group])
            data[combination] = data[combination][samples_from_each_group:]

    return nodes, data


def create_unfair_nodes(
    fair_nodes: list,
    nodes_to_unfair: list,
    remaining_data: dict,
    group_to_reduce: tuple,
    group_to_increment: tuple,
    ratio_unfairness: tuple,
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
            if (sample["y"], sample["z"]) == group_to_reduce:
                count_sensitive_group_samples += 1

        # We compute the number of samples that we want to remove from the group_to_reduce
        # based on the ratio_unfairness
        current_ratio = np.random.uniform(ratio_unfairness[0], ratio_unfairness[1])
        samples_to_be_removed = int(count_sensitive_group_samples * current_ratio)
        number_of_samples_to_add.append(samples_to_be_removed)

        for sample in node:
            # Now we remove the samples from the group_to_reduce
            # and we store them in removed_samples
            if (
                sample["y"],
                sample["z"],
            ) == group_to_reduce and samples_to_be_removed > 0:
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
                    sample["y"],
                    sample["z"],
                ) == group_to_increment and samples_to_remove > 0:
                    if (sample["y"], sample["z"]) not in remaining_data:
                        remaining_data[group_to_increment] = []
                    remaining_data[group_to_increment].append(sample)
                    samples_to_remove -= 1
                    node.pop(index)
            if sum(number_of_samples_to_add) > 0:
                assert samples_to_remove == 0, "Not enough samples to remove"
        assert sum(number_of_samples_to_add) <= len(
            remaining_data[group_to_increment]
        ), "Too many samples to add"
        # now we have to add the same amount of data taken from group_to_unfair
        for node, samples_to_add in zip(unfair_nodes, number_of_samples_to_add):
            node.extend(remaining_data[group_to_increment][:samples_to_add])
            remaining_data[group_to_increment] = remaining_data[group_to_increment][
                samples_to_add:
            ]

    return fair_nodes, unfair_nodes


def representative_diversity_approach(X, y, z, num_nodes, number_of_samples_per_node):
    """
    With this approach we want to distribute the data among the nodes in a representative diversity way.
    This means that each node has the same ratio of each group that we are observing in the dataset

    params:
    X: numpy array of shape (N, D) where N is the number of samples and D is the number of features
    y: numpy array of shape (N, ) where N is the number of samples. Here we have the samples labels
    z: numpy array of shape (N, ) where N is the number of samples. Here we have the samples sensitive features
    num_nodes: number of nodes to generate
    number_of_samples_per_node: number of samples that we want in each node. Can be None, in this case we just use
        len(y)//num_nodes
    """
    samples_per_node = (
        number_of_samples_per_node
        if number_of_samples_per_node
        else len(y) // num_nodes
    )
    # create the nodes sampling from the dataset wihout replacement
    dataset = [{"x": x_, "y": y_, "z": z_} for x_, y_, z_ in zip(X, y, z)]
    # shuffle the dataset
    np.random.shuffle(dataset)

    # Distribute the data among the nodes with a random sample from the dataset
    # considering the number of samples per node
    nodes = []
    for i in range(num_nodes):
        nodes.append([])
        nodes[i].extend(dataset[:samples_per_node])
        dataset = dataset[samples_per_node:]

    # Create the dictionary with the remaining data
    remaining_data = {}
    for sample in dataset:
        if (sample["y"], sample["z"]) not in remaining_data:
            remaining_data[(sample["y"], sample["z"])] = []
        remaining_data[(sample["y"], sample["z"])].append(sample)

    return nodes, remaining_data


def generate_clients_biased_data_mod(
    X,
    y,
    z,
    approach: str,
    num_nodes: int,
    ratio_unfair_nodes: float,
    opposite_direction: bool,
    ratio_unfairness: tuple,
    group_to_reduce: tuple = None,
    group_to_increment: tuple = None,
    number_of_samples_per_node: int = None,
    opposite_group_to_reduce: tuple = None,
    opposite_group_to_increment: tuple = None,
    opposite_ratio_unfairness: tuple = None,
    one_group_nodes: bool = False,
):
    """
    This function generates the data for the clients.

    params:
    X: numpy array of shape (N, D) where N is the number of samples and D is the number of features
    y: numpy array of shape (N, ) where N is the number of samples. Here we have the samples labels
    z: numpy array of shape (N, ) where N is the number of samples. Here we have the samples sensitive features
    num_nodes: number of nodes to generate
    approach: type of approach we want to use to distribute the data among the fair clients. This can be egalitarian or representative
    ratio_unfair_nodes: the fraction of unfair clients we want to have in the experiment
    opposite_direction: true if we want to allow different nodes to have different majoritiarian classes. For instance,
        we could have some nodes with a max disparity that depends on the majority class being 0 and other nodes with a max disparity
        that depends on the majority class being 1.
    group_to_reduce: the group that we want to be unfair. For instance, in the case of binary target and binary sensitive value
        we could have (0,0), (0,1), (1,0) or (1,1)
    ratio_unfairness: tuple (min, max) where min is the minimum ratio of samples that we want to remove from the group_to_reduce
        and max is the maximum ratio of samples that we want to remove from the group_to_reduce
    """

    # check if the number of samples that we want in each node is
    # greater than the number of samples we have in the dataset
    if number_of_samples_per_node:
        assert (
            number_of_samples_per_node < len(y) // num_nodes
        ), "Too many samples per node"
    # check if the ratio_fair_nodes is between 0 and 1
    assert ratio_unfair_nodes <= 1, "ratio_unfair_nodes must be less or equal than 1"
    assert ratio_unfair_nodes >= 0, "ratio_unfair_nodes must be greater or equal than 0"
    assert group_to_reduce, "group_to_reduce must be specified"
    assert group_to_increment, "group_to_increment must be specified"
    # check if the approach type is egalitarian or representative
    assert approach in [
        "egalitarian",
        "representative",
    ], "Approach must be egalitarian or representative"

    number_unfair_nodes = int(num_nodes * ratio_unfair_nodes)
    number_fair_nodes = num_nodes - number_unfair_nodes
    if approach == "egalitarian":
        # first split the data among the nodes in an egalitarian way
        # each node has the same amount of data and the same ratio of each group
        nodes, remaining_data = egalitarian_approach(
            X, y, z, num_nodes, number_of_samples_per_node
        )
    else:
        nodes, remaining_data = representative_diversity_approach(
            X, y, z, num_nodes, number_of_samples_per_node
        )

    if opposite_direction:
        assert opposite_group_to_reduce, "opposite_group_to_reduce must be specified"
        assert (
            opposite_group_to_increment
        ), "opposite_group_to_increment must be specified"
        group_size = number_unfair_nodes // 2
        unfair_nodes_direction_1 = create_unfair_nodes(
            nodes_to_unfair=nodes[number_fair_nodes : number_fair_nodes + group_size],
            remaining_data=remaining_data,
            group_to_reduce=group_to_reduce,
            group_to_increment=group_to_increment,
            ratio_unfairness=ratio_unfairness,
        )
        unfair_nodes_direction_2 = create_unfair_nodes(
            nodes_to_unfair=nodes[number_fair_nodes + group_size :],
            remaining_data=remaining_data,
            group_to_reduce=opposite_group_to_reduce,
            group_to_increment=opposite_group_to_increment,
            ratio_unfairness=opposite_ratio_unfairness,
        )
        return (
            nodes[0:number_fair_nodes]
            + unfair_nodes_direction_1
            + unfair_nodes_direction_2
        ), [0] * number_fair_nodes + [1] * len(unfair_nodes_direction_1)
    else:
        # At the moment this is the only thing that is working, we need
        # to fix the opposite direction version
        fair_nodes, unfair_nodes = create_unfair_nodes(
            fair_nodes=nodes[:number_fair_nodes],
            nodes_to_unfair=nodes[number_fair_nodes:],
            remaining_data=remaining_data,
            group_to_reduce=group_to_reduce,
            group_to_increment=group_to_increment,
            ratio_unfairness=ratio_unfairness,
        )

        if one_group_nodes:
            # create the nodes that only have one group
            fair_nodes, unfair_nodes = create_one_group_nodes(
                fair_nodes, unfair_nodes, ratio_unfair_nodes
            )
        return (
            fair_nodes + unfair_nodes,
            [0] * number_fair_nodes + [1] * number_fair_nodes,
        )


def create_one_group_nodes(fair_nodes, unfair_nodes, ratio_unfair_nodes):
    # num_one_group_nodes = int(
    #     (len(fair_nodes) + len(unfair_nodes)) * ratio_one_group_nodes
    # )
    num_one_group_nodes_fair = len(
        fair_nodes
    )  # int(num_one_group_nodes * (1 - ratio_unfair_nodes))
    # if num_one_group_nodes_fair % 2 != 0:
    #     num_one_group_nodes_fair = num_one_group_nodes_fair - 1
    num_one_group_nodes_unfair = len(
        unfair_nodes
    )  # num_one_group_nodes - num_one_group_nodes_fair
    # if num_one_group_nodes_unfair % 2 != 0:
    #     num_one_group_nodes_unfair = num_one_group_nodes_unfair - 1

    # modified_nodes = []
    removed_samples = {"0": [], "1": []}
    number_removed_samples = {}

    # Remove samples from the fair nodes and from the unfair nodes

    tmp_fair_nodes = []
    for node_id, node in enumerate(fair_nodes[:num_one_group_nodes_fair]):
        tmp_removed_samples = []
        tmp_samples = []
        for sample in node:
            # if node_id % 2 == 0:
            #     if sample["z"] == 0:
            #         tmp_removed_samples.append(sample)
            #     else:
            #         tmp_samples.append(sample)
            # else:
            if sample["z"] == 1 and node_id % 2 == 0:
                tmp_removed_samples.append(sample)
            else:
                tmp_samples.append(sample)

        tmp_fair_nodes.append(tmp_samples)
        removed_samples[str(node_id % 2)].extend(tmp_removed_samples)
        number_removed_samples[node_id] = len(tmp_removed_samples)

    return tmp_fair_nodes, unfair_nodes


def load_dutch(dataset_path):
    data = arff.loadarff(dataset_path + "dutch_census.arff")
    dutch_df = pd.DataFrame(data[0]).astype("int32")
    # dutch_df = pd.read_csv(dataset_path + "dutch_census_removed.csv")

    dutch_df["sex_binary"] = np.where(dutch_df["sex"] == 1, 1, 0)
    dutch_df["occupation_binary"] = np.where(dutch_df["occupation"] >= 300, 1, 0)

    del dutch_df["sex"]
    del dutch_df["occupation"]

    dutch_df_feature_columns = [
        "age",
        "household_position",
        "household_size",
        "prev_residence_place",
        "citizenship",
        "country_birth",
        "edu_level",
        "economic_status",
        "cur_eco_activity",
        "Marital_status",
        "sex_binary",
    ]

    metadata_dutch = {
        "name": "Dutch census",
        "code": ["DU1"],
        "protected_atts": ["sex_binary"],
        "protected_att_values": [0],
        "protected_att_descriptions": ["Gender = Female"],
        "target_variable": "occupation_binary",
    }

    return dutch_df, dutch_df_feature_columns, metadata_dutch


## Use this function to retrieve X, X, y arrays for training ML models
def dataset_to_numpy(
    _df,
    _feature_cols: list,
    _metadata: dict,
    num_sensitive_features: int = 1,
    sensitive_features_last: bool = True,
):
    """Args:
    _df: pandas dataframe
    _feature_cols: list of feature column names
    _metadata: dictionary with metadata
    num_sensitive_features: number of sensitive features to use
    sensitive_features_last: if True, then sensitive features are encoded as last columns
    """

    # transform features to 1-hot
    print(_feature_cols)
    print(_df.columns)
    _X = _df[_feature_cols]
    # take sensitive features separately
    print(
        f'Using {_metadata["protected_atts"][:num_sensitive_features]} as sensitive feature(s).'
    )
    if num_sensitive_features > len(_metadata["protected_atts"]):
        num_sensitive_features = len(_metadata["protected_atts"])
    _Z = _X[_metadata["protected_atts"][:num_sensitive_features]]
    _X = _X.drop(columns=_metadata["protected_atts"][:num_sensitive_features])

    # 1-hot encode and scale features
    if "dummy_cols" in _metadata.keys():
        dummy_cols = _metadata["dummy_cols"]
    else:
        dummy_cols = None
    _X2 = pd.get_dummies(_X, columns=dummy_cols, drop_first=False)
    esc = MinMaxScaler()
    _X = esc.fit_transform(_X2)

    # original
    # current implementation assumes each sensitive feature is binary
    for i, tmp in enumerate(_metadata["protected_atts"][:num_sensitive_features]):
        assert len(_Z[tmp].unique()) == 2, "Sensitive feature is not binary!"

    # 1-hot sensitive features, (optionally) swap ordering so privileged class feature == 1 is always last, preceded by the corresponding unprivileged feature
    _Z2 = pd.get_dummies(_Z, columns=_Z.columns, drop_first=False)
    # print(_Z2.head(), _Z2.shape)
    if sensitive_features_last:
        for i, tmp in enumerate(_Z.columns):
            assert (
                _metadata["protected_att_values"][i] in _Z[tmp].unique()
            ), "Protected attribute value not found in data!"
            if not np.allclose(float(_metadata["protected_att_values"][i]), 0):
                # swap columns
                _Z2.iloc[:, [2 * i, 2 * i + 1]] = _Z2.iloc[:, [2 * i + 1, 2 * i]]
    # change booleans to floats
    # _Z2 = _Z2.astype(float)

    # original
    _Z = _Z2.to_numpy()

    # _Z = _Z.to_numpy()

    _y = _df[_metadata["target_variable"]].values
    return _X, _Z, _y


# Use this function to retrieve X, X, y arrays for training ML models
def dataset_to_numpy_mod(
    _df,
    _feature_cols: list,
    _metadata: dict,
    num_sensitive_features: int = 1,
    sensitive_features_last: bool = True,
):
    """Args:
    _df: pandas dataframe
    _feature_cols: list of feature column names
    _metadata: dictionary with metadata
    num_sensitive_features: number of sensitive features to use
    sensitive_features_last: if True, then sensitive features are encoded as last columns
    """

    # transform features to 1-hot
    print(_feature_cols)

    _X = _df[_feature_cols]
    # take sensitive features separately
    print(
        f'Using {_metadata["protected_atts"][:num_sensitive_features]} as sensitive feature(s).'
    )
    if num_sensitive_features > len(_metadata["protected_atts"]):
        num_sensitive_features = len(_metadata["protected_atts"])
    _Z = _X[_metadata["protected_atts"][:num_sensitive_features]]
    # _X = _X.drop(columns=_metadata["protected_atts"][:num_sensitive_features])

    my_sensitive_features = _X[["edu_level"]]

    # 1-hot encode and scale features
    if "dummy_cols" in _metadata.keys():
        dummy_cols = _metadata["dummy_cols"]
    else:
        dummy_cols = None
    _X2 = pd.get_dummies(_X, columns=dummy_cols, drop_first=False)
    esc = MinMaxScaler()
    _X = esc.fit_transform(_X2)

    # original
    # current implementation assumes each sensitive feature is binary
    for i, tmp in enumerate(_metadata["protected_atts"][:num_sensitive_features]):
        assert len(_Z[tmp].unique()) == 2, "Sensitive feature is not binary!"

    # 1-hot sensitive features, (optionally) swap ordering so privileged class feature == 1 is always last, preceded by the corresponding unprivileged feature
    _Z2 = pd.get_dummies(_Z, columns=_Z.columns, drop_first=False)
    # print(_Z2.head(), _Z2.shape)
    if sensitive_features_last:
        for i, tmp in enumerate(_Z.columns):
            assert (
                _metadata["protected_att_values"][i] in _Z[tmp].unique()
            ), "Protected attribute value not found in data!"
            if not np.allclose(float(_metadata["protected_att_values"][i]), 0):
                # swap columns
                _Z2.iloc[:, [2 * i, 2 * i + 1]] = _Z2.iloc[:, [2 * i + 1, 2 * i]]
    # change booleans to floats
    # _Z2 = _Z2.astype(float)

    # original
    _Z = _Z2.to_numpy()

    # _Z = _Z.to_numpy()
    _Z = my_sensitive_features.to_numpy()
    print(_Z)

    _y = _df[_metadata["target_variable"]].values
    return _X, _Z, _y


def get_tabular_numpy_dataset(dataset_name, num_sensitive_features, dataset_path=None):
    if dataset_name == "dutch":
        tmp = load_dutch(dataset_path=dataset_path)
    else:
        raise ValueError("Unknown dataset name!")
    _X, _Z, _y = dataset_to_numpy(*tmp, num_sensitive_features=num_sensitive_features)
    return _X, _Z, _y


def prepare_tabular_data(
    dataset_path: str,
    dataset_name: str,
    approach: str,
    num_nodes: int,
    ratio_unfair_nodes: float,
    opposite_direction: bool,
    ratio_unfairness: tuple,
    group_to_reduce: tuple = None,
    group_to_increment: tuple = None,
    number_of_samples_per_node: int = None,
    opposite_group_to_reduce: tuple = None,
    opposite_group_to_increment: tuple = None,
    opposite_ratio_unfairness: tuple = None,
    do_iid_split: bool = False,
    one_group_nodes: bool = False,
    splitted_data_dir: str = None,
):
    if dataset_name == "income":
        for client_name in range(num_nodes):
            os.system(
                f"rm -rf {dataset_path}/{splitted_data_dir}/{client_name}/train.pt"
            )

            # open numpy arrays
            X = np.load(
                f"{dataset_path}/{splitted_data_dir}/{client_name}/income_dataframes_{client_name}.npy"
            )
            Y = np.load(
                f"{dataset_path}/{splitted_data_dir}/{client_name}/income_labels_{client_name}.npy"
            )
            Z = np.load(
                f"{dataset_path}/{splitted_data_dir}/{client_name}/income_groups_{client_name}.npy"
            )
            custom_dataset = TabularDataset(
                x=np.hstack((X, np.ones((X.shape[0], 1)))).astype(np.float32),
                z=[item.item() for item in Z],  # .astype(np.float32),
                y=[item.item() for item in Y],  # .astype(np.float32),
            )
            torch.save(
                custom_dataset,
                f"{dataset_path}/{splitted_data_dir}/{client_name}/train.pt",
            )

        fed_dir = f"{dataset_path}/{splitted_data_dir}"
        return fed_dir, None
    else:
        # client_data, N_is, props_positive = get_tabular_data(
        client_data, disparities, metadata = get_tabular_data(
            num_clients=num_nodes,
            do_iid_split=do_iid_split,
            dataset_name=dataset_name,
            num_sensitive_features=1,
            dataset_path=dataset_path,
            approach=approach,
            num_nodes=num_nodes,
            ratio_unfair_nodes=ratio_unfair_nodes,
            opposite_direction=opposite_direction,
            ratio_unfairness=ratio_unfairness,
            group_to_reduce=group_to_reduce,
            group_to_increment=group_to_increment,
            number_of_samples_per_node=number_of_samples_per_node,
            opposite_group_to_reduce=opposite_group_to_reduce,
            opposite_group_to_increment=opposite_group_to_increment,
            opposite_ratio_unfairness=opposite_ratio_unfairness,
            one_group_nodes=one_group_nodes,
        )

        # transform client data so that they are compatiblw with the
        # other functions
        tmp_data = []
        possible_z = np.array([])
        possible_y = np.array([])
        for client in client_data:
            tmp_x = []
            tmp_y = []
            tmp_z = []
            for sample in client:
                tmp_x.append(sample["x"])
                tmp_y.append(sample["y"])
                tmp_z.append(sample["z"])

            tmp_data.append(
                {"x": np.array(tmp_x), "y": np.array(tmp_y), "z": np.array(tmp_z)}
            )
            unique_z = np.unique(np.array(tmp_z))
            unique_y = np.unique(np.array(tmp_y))
            possible_z = np.unique(np.concatenate((possible_z, unique_z)))
            possible_y = np.unique(np.concatenate((possible_y, unique_y)))
        client_data = tmp_data

        predictions = []
        sensitive_features = []

        # remove the old files in the data folder
        os.system(f"rm -rf {dataset_path}/{splitted_data_dir}/*")
        for client_name, (client, client_disparity, client_metadata) in enumerate(
            zip(client_data, disparities, metadata)
        ):
            # Append 1 to each samples

            custom_dataset = TabularDataset(
                x=np.hstack((client["x"], np.ones((client["x"].shape[0], 1)))).astype(
                    np.float32
                ),
                z=client["z"],  # .astype(np.float32),
                y=client["y"],  # .astype(np.float32),
            )
            # Create the folder for the user client_name
            os.system(f"mkdir {dataset_path}/{splitted_data_dir}/{client_name}")
            # store the dataset in the client folder with the name "train.pt"
            torch.save(
                custom_dataset,
                f"{dataset_path}/{splitted_data_dir}/{client_name}/train.pt",
            )
            # store statistics about the dataset in the same folder
            statistics = Utils.get_dataset_statistics(
                custom_dataset, client_disparity, client_metadata
            )
            with open(
                f"{dataset_path}/{splitted_data_dir}/{client_name}/metadata.json", "w"
            ) as outfile:
                print(statistics)
                json_object = json.dumps(statistics, indent=4)
                outfile.write(json_object)

            predictions.append(list(client["y"]))
            sensitive_features.append(list(client["z"]))

        counter_distribution_nodes = Utils.compute_distribution_debug(
            predictions=predictions, sensitive_features=sensitive_features
        )

        possible_y = [str(int(item)) for item in possible_y.tolist()]
        possible_z = [str(int(item)) for item in possible_z.tolist()]
        # we are still assuming a binary target
        # however, we can have a non binary sensitive value
        missing_combinations = []
        all_combinations = []
        sent_disparity_combinations = [f"1|{sensitive}" for sensitive in possible_z]
        for combination in sent_disparity_combinations:
            missing_combinations.append(("0" + combination[1:], combination))
            all_combinations.append(combination)
            all_combinations.append("0" + combination[1:])

        fed_dir = f"{dataset_path}/{splitted_data_dir}"
        json_file = {
            "possible_z": possible_z,
            "possible_y": possible_y,
            "missing_combinations": missing_combinations,
            "all_combinations": all_combinations,
            "combinations": sent_disparity_combinations,
        }
        with open(f"{fed_dir}/metadata.json", "w") as outfile:
            json_object = json.dumps(json_file, indent=4)
            outfile.write(json_object)

        Utils.plot_distributions(
            title="Distribution of the nodes",
            counter_groups=counter_distribution_nodes,
            nodes=[f"{i}" for i in range(len(counter_distribution_nodes))],
            all_combinations=all_combinations,
        )

        return fed_dir, client_data
