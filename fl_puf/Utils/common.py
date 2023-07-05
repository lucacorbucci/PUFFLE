# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Commonly used functions for generating partitioned datasets."""

# pylint: disable=invalid-name
import random
from collections import Counter
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.random import BitGenerator, Generator, SeedSequence

XYZ = Tuple[np.ndarray, np.ndarray]
XYZList = List[XYZ]
PartitionedDataset = Tuple[XYZList, XYZList]

np.random.seed(2020)


def float_to_int(i: float) -> int:
    """Return float as int but raise if decimal is dropped."""
    if not i.is_integer():
        raise Exception("Cast would drop decimals")

    return int(i)


def sort_by_label(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> XYZ:
    """Sort by label.

    Assuming two labels and four examples the resulting label order
    would be 1,1,2,2
    """
    idx = np.argsort(z, axis=0).reshape((z.shape[0]))
    return (x[idx], y[idx], z[idx])


def sort_by_sensitive_value(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> XYZ:
    """Sort by label.

    Assuming two labels and four examples the resulting label order
    would be 1,1,2,2
    """
    idx = np.argsort(y, axis=0).reshape((y.shape[0]))
    return (x[idx], y[idx], z[idx])


def sort_by_label_repeating(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> XYZ:
    """Sort by label in repeating groups. Assuming two labels and four examples
    the resulting label order would be 1,2,1,2.

    Create sorting index which is applied to by label sorted x, y

    .. code-block:: python

        # given:
        y = [
            0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9
        ]

        # use:
        idx = [
            0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19
        ]

        # so that y[idx] becomes:
        y = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
        ]
    """
    x, y, z = sort_by_label(x, y, z)

    num_example = x.shape[0]
    num_class = np.unique(z).shape[0]
    idx = (
        np.array(range(num_example), np.int64)
        .reshape((num_class, num_example // num_class))
        .transpose()
        .reshape(num_example)
    )

    return (x[idx], y[idx], z[idx])


def split_at_fraction(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, fraction: float
) -> Tuple[XYZ, XYZ]:
    """Split x, y at a certain fraction."""
    splitting_index = float_to_int(x.shape[0] * fraction)
    # Take everything BEFORE splitting_index
    x_0, y_0, z_0 = x[:splitting_index], y[:splitting_index], z[:splitting_index]
    # Take everything AFTER splitting_index
    x_1, y_1, z_1 = x[splitting_index:], y[splitting_index:], z[splitting_index:]
    return (x_0, y_0, z_0), (x_1, y_1, z_1)


def shuffle(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> XYZ:
    """Shuffle x and y."""
    idx = np.random.permutation(len(x))
    return x[idx], y[idx], z[idx]


def partition(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, num_partitions: int
) -> List[XYZ]:
    """Return x, y as list of partitions."""
    return list(
        zip(
            np.split(x, num_partitions),
            np.split(y, num_partitions),
            np.split(z, num_partitions),
        ),
    )


def combine_partitions(XYZ_list_0: XYZList, XYZ_list_1: XYZList) -> XYZList:
    """Combine two lists of ndarray Tuples into one list."""
    return [
        (
            np.concatenate([x_0, x_1], axis=0),
            np.concatenate([y_0, y_1], axis=0),
            np.concatenate([z_0, z_1], axis=0),
        )
        for (x_0, y_0, z_0), (x_1, y_1, z_1) in zip(XYZ_list_0, XYZ_list_1)
    ]


def shift(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> XYZ:
    """Shift x_1, y_1 so that the first half contains only labels 0 to 4 and
    the second half 5 to 9."""
    x, y, z = sort_by_label(x, y, z)

    (x_0, y_0, z_0), (x_1, y_1, z_1) = split_at_fraction(x, y, z, fraction=0.5)
    (x_0, y_0, z_0), (x_1, y_1, z_1) = shuffle(x_0, y_0, z_0), shuffle(x_1, y_1, z_1)
    x, y, z = (
        np.concatenate([x_0, x_1], axis=0),
        np.concatenate([y_0, y_1], axis=0),
        np.concatenate([z_0, z_1], axis=0),
    )
    return x, y, z


def create_partitions(
    unpartitioned_dataset: XYZ,
    iid_fraction: float,
    num_partitions: int,
) -> XYZList:
    """Create partitioned version of a training or test set.

    Currently tested and supported are MNIST, FashionMNIST and
    CIFAR-10/100
    """
    x, y, z = unpartitioned_dataset

    x, y, z = shuffle(x, y, z)
    x, y, z = sort_by_label_repeating(x, y, z)

    (x_0, y_0, z_0), (x_1, y_1, z_1) = split_at_fraction(x, y, z, fraction=iid_fraction)

    # Shift in second split of dataset the classes into two groups
    x_1, y_1, z_1 = shift(x_1, y_1, z_1)

    XYZ_0_partitions = partition(x_0, y_0, z_0, num_partitions)
    XYZ_1_partitions = partition(x_1, y_1, z_1, num_partitions)

    XYZ_partitions = combine_partitions(XYZ_0_partitions, XYZ_1_partitions)

    # Adjust x and y shape
    return [adjust_XYZ_shape(XYZ) for XYZ in XYZ_partitions]


def create_partitioned_dataset(
    keras_dataset: Tuple[XYZ, XYZ],
    iid_fraction: float,
    num_partitions: int,
) -> Tuple[PartitionedDataset, XYZ]:
    """Create partitioned version of keras dataset.

    Currently tested and supported are MNIST, FashionMNIST and
    CIFAR-10/100
    """
    XYZ_train, XYZ_test = keras_dataset

    XYZ_train_partitions = create_partitions(
        unpartitioned_dataset=XYZ_train,
        iid_fraction=iid_fraction,
        num_partitions=num_partitions,
    )

    XYZ_test_partitions = create_partitions(
        unpartitioned_dataset=XYZ_test,
        iid_fraction=iid_fraction,
        num_partitions=num_partitions,
    )

    return (XYZ_train_partitions, XYZ_test_partitions), adjust_XYZ_shape(XYZ_test)


def log_distribution(XYZ_partitions: XYZList) -> None:
    """Print label distribution for list of paritions."""
    distro = [np.unique(z, return_counts=True) for _, _, z in XYZ_partitions]
    for d in distro:
        print(d)


def adjust_XYZ_shape(XYZ: XYZ) -> XYZ:
    """Adjust shape of both x and y."""
    x, y, z = XYZ
    if x.ndim == 3:
        x = adjust_x_shape(x)
    if z.ndim == 2:
        z = adjust_y_shape(z)
    return (x, y, z)


def adjust_x_shape(nda: np.ndarray) -> np.ndarray:
    """Turn shape (x, y, z) into (x, y, z, 1)."""
    nda_adjusted = np.reshape(nda, (nda.shape[0], nda.shape[1], nda.shape[2], 1))
    return nda_adjusted


def adjust_y_shape(nda: np.ndarray) -> np.ndarray:
    """Turn shape (x, 1) into (x)."""
    nda_adjusted = np.reshape(nda, (nda.shape[0]))
    return nda_adjusted


def split_array_at_indices(
    x: np.ndarray, split_idx: np.ndarray
) -> List[List[np.ndarray]]:
    """Splits an array `x` into list of elements using starting indices from
    `split_idx`.

        This function should be used with `unique_indices` from `np.unique()` after
        sorting by label.

    Args:
        x (np.ndarray): Original array of dimension (N,a,b,c,...)
        split_idx (np.ndarray): 1-D array contaning increasing number of
            indices to be used as partitions. Initial value must be zero. Last value
            must be less than N.

    Returns:
        List[List[np.ndarray]]: List of list of samples.
    """

    if split_idx.ndim != 1:
        raise ValueError("Variable `split_idx` must be a 1-D numpy array.")
    if split_idx.dtype != np.int64:
        raise ValueError("Variable `split_idx` must be of type np.int64.")
    if split_idx[0] != 0:
        raise ValueError("First value of `split_idx` must be 0.")
    if split_idx[-1] >= x.shape[0]:
        raise ValueError(
            """Last value in `split_idx` must be less than
            the number of samples in `x`."""
        )
    if not np.all(split_idx[:-1] <= split_idx[1:]):
        raise ValueError("Items in `split_idx` must be in increasing order.")

    num_splits: int = len(split_idx)
    split_idx = np.append(split_idx, x.shape[0])

    list_samples_split: List[List[np.ndarray]] = [[] for _ in range(num_splits)]
    for j in range(num_splits):
        tmp_x = x[split_idx[j] : split_idx[j + 1]]  # noqa: E203
        for sample in tmp_x:
            list_samples_split[j].append(sample)

    return list_samples_split


def exclude_classes_and_normalize(
    distribution: np.ndarray, exclude_dims: List[bool], eps: float = 1e-5
) -> np.ndarray:
    """Excludes classes from a distribution.

    This function is particularly useful when sampling without replacement.
    Classes for which no sample is available have their probabilities are set to 0.
    Classes that had probabilities originally set to 0 are incremented with
     `eps` to allow sampling from remaining items.

    Args:
        distribution (np.array): Distribution being used.
        exclude_dims (List[bool]): Dimensions to be excluded.
        eps (float, optional): Small value to be addad to non-excluded dimensions.
            Defaults to 1e-5.

    Returns:
        np.ndarray: Normalized distributions.
    """
    if np.any(distribution < 0) or (not np.isclose(np.sum(distribution), 1.0)):
        raise ValueError("distribution must sum to 1 and have only positive values.")

    if distribution.size != len(exclude_dims):
        raise ValueError(
            """Length of distribution must be equal
            to the length `exclude_dims`."""
        )
    if eps < 0:
        raise ValueError("""The value of `eps` must be positive and small.""")

    distribution[[not x for x in exclude_dims]] += eps
    distribution[exclude_dims] = 0.0
    sum_rows = np.sum(distribution) + np.finfo(float).eps
    distribution = distribution / sum_rows

    return distribution


def sample_without_replacement(
    distribution: np.ndarray,
    list_samples: List[List[np.ndarray]],
    list_sensitive_features_per_class: List[List[np.ndarray]],
    num_samples: int,
    empty_classes: List[bool],
) -> Tuple[XYZ, List[bool]]:
    """Samples from a list without replacement using a given distribution.

    Args:
        distribution (np.ndarray): Distribution used for sampling.
        list_samples(List[List[np.ndarray]]): List of samples.
        num_samples (int): Total number of items to be sampled.
        empty_classes (List[bool]): List of booleans indicating which classes are empty.
            This is useful to differentiate which classes should still be sampled.

    Returns:
        XYZ: Dataset contaning samples
        List[bool]: empty_classes.
    """
    if np.sum([len(x) for x in list_samples]) < num_samples:
        raise ValueError(
            """Number of samples in `list_samples` is less than `num_samples`"""
        )

    # Make sure empty classes are not sampled
    # and solves for rare cases where
    if not empty_classes:
        empty_classes = len(distribution) * [False]

    distribution = exclude_classes_and_normalize(
        distribution=distribution, exclude_dims=empty_classes
    )

    data: List[np.ndarray] = []
    target: List[np.ndarray] = []
    sensitive_list: List[np.ndarray] = []

    # check this or find a different dirty solution to run an experiment
    for _ in range(num_samples):
        sample_class = np.where(np.random.multinomial(1, distribution) == 1)[0][0]
        sample: np.ndarray = list_samples[sample_class].pop()
        sensitive_feature = list_sensitive_features_per_class[sample_class].pop()

        data.append(sample)
        target.append(sample_class)
        sensitive_list.append(sensitive_feature)

        # If last sample of the class was drawn, then set the
        #  probability density function (PDF) to zero for that class.
        if len(list_samples[sample_class]) == 0:
            empty_classes[sample_class] = True
            # Be careful to distinguish between classes that had zero probability
            # and classes that are now empty
            distribution = exclude_classes_and_normalize(
                distribution=distribution, exclude_dims=empty_classes
            )
    data_array: np.ndarray = np.concatenate([data], axis=0)
    target_array: np.ndarray = np.array(target, dtype=np.int64)
    sensitive_array: np.ndarray = np.array(sensitive_list, dtype=np.int64)

    return (data_array, sensitive_array, target_array), empty_classes


def sample_without_replacement_sensitive(
    distribution: np.ndarray,
    list_samples_per_sensitive_feature: List[List[np.ndarray]],
    list_class_per_sensitive_feature: List[List[np.ndarray]],
    num_samples: int,
    empty_classes: List[bool],
) -> Tuple[XYZ, List[bool]]:
    """Samples from a list without replacement using a given distribution.

    Args:
        distribution (np.ndarray): Distribution used for sampling.
        list_samples(List[List[np.ndarray]]): List of samples.
        num_samples (int): Total number of items to be sampled.
        empty_classes (List[bool]): List of booleans indicating which classes are empty.
            This is useful to differentiate which classes should still be sampled.

    Returns:
        XYZ: Dataset contaning samples
        List[bool]: empty_classes.
    """
    import random

    if np.sum([len(x) for x in list_samples_per_sensitive_feature]) < num_samples:
        raise ValueError(
            """Number of samples in `list_samples` is less than `num_samples`"""
        )

    # Make sure empty classes are not sampled
    # and solves for rare cases where
    if not empty_classes:
        empty_classes = len(distribution) * [False]

    distribution = exclude_classes_and_normalize(
        distribution=distribution, exclude_dims=empty_classes
    )

    data: List[np.ndarray] = []
    target: List[np.ndarray] = []
    sensitive_list: List[np.ndarray] = []

    # check this or find a different dirty solution to run an experiment
    for _ in range(num_samples):
        sample_sensitive_feature = np.where(
            np.random.multinomial(1, distribution) == 1
        )[0][0]
        sample: np.ndarray = list_samples_per_sensitive_feature[
            sample_sensitive_feature
        ].pop()
        classes = list_class_per_sensitive_feature[sample_sensitive_feature].pop()

        data.append(sample)
        sensitive_list.append(sample_sensitive_feature)
        target.append(classes)

        # If last sample of the class was drawn, then set the
        #  probability density function (PDF) to zero for that class.
        if len(list_samples_per_sensitive_feature[sample_sensitive_feature]) == 0:
            empty_classes[sample_sensitive_feature] = True
            # Be careful to distinguish between classes that had zero probability
            # and classes that are now empty
            distribution = exclude_classes_and_normalize(
                distribution=distribution, exclude_dims=empty_classes
            )
    data_array: np.ndarray = np.concatenate([data], axis=0)
    target_array: np.ndarray = np.array(target, dtype=np.int64)
    sensitive_array: np.ndarray = np.array(sensitive_list, dtype=np.int64)

    return (data_array, sensitive_array, target_array), empty_classes


def get_partitions_distributions(partitions: XYZList) -> Tuple[np.ndarray, List[int]]:
    """Evaluates the distribution over classes for a set of partitions.

    Args:
        partitions (XYZList): Input partitions

    Returns:
        np.ndarray: Distributions of size (num_partitions, num_classes)
    """
    # Get largest available label
    labels = set()
    for _, _, z in partitions:
        labels.update(set(z))
    list_labels = sorted(list(labels))
    bin_edges = np.arange(len(list_labels) + 1)

    # Pre-allocate distributions
    distributions = np.zeros((len(partitions), len(list_labels)), dtype=np.float32)
    for idx, (_, _, _z) in enumerate(partitions):
        hist, _ = np.histogram(_z, bin_edges)
        distributions[idx] = hist / hist.sum()

    return distributions, list_labels


def create_lda_partitions(
    dataset: XYZ,
    dirichlet_dist: Optional[np.ndarray] = None,
    num_partitions: int = 100,
    concentration: Union[float, np.ndarray, List[float]] = 0.5,
    accept_imbalanced: bool = False,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> Tuple[XYZList, np.ndarray]:
    """Create imbalanced non-iid partitions using Latent Dirichlet Allocation
    (LDA) without resampling.

    Args:
        dataset (XYZ): Dataset containing samples X and labels Y.
        dirichlet_dist (numpy.ndarray, optional): previously generated distribution to
            be used. This is useful when applying the same distribution for train and
            validation sets.
        num_partitions (int, optional): Number of partitions to be created.
            Defaults to 100.
        concentration (float, np.ndarray, List[float]): Dirichlet Concentration
            (:math:`\\alpha`) parameter. Set to float('inf') to get uniform partitions.
            An :math:`\\alpha \\to \\Inf` generates uniform distributions over classes.
            An :math:`\\alpha \\to 0.0` generates one class per client. Defaults to 0.5.
        accept_imbalanced (bool): Whether or not to accept imbalanced output classes.
            Default False.
        seed (None, int, SeedSequence, BitGenerator, Generator):
            A seed to initialize the BitGenerator for generating the Dirichlet
            distribution. This is defined in Numpy's official documentation as follows:
            If None, then fresh, unpredictable entropy will be pulled from the OS.
            One may also pass in a SeedSequence instance.
            Additionally, when passed a BitGenerator, it will be wrapped by Generator.
            If passed a Generator, it will be returned unaltered.
            See official Numpy Documentation for further details.

    Returns:
        Tuple[XYZList, numpy.ndarray]: List of XYZList containing partitions
            for each dataset and the dirichlet probability density functions.
            The returned list contains N elements where N is the number of clients
            and each element is a tuple containing the following:
            - position 0 -> the samples for the client
            - position 1 -> the sensitive features for the client
            - position 2 -> the labels for the client
    """
    # pylint: disable=too-many-arguments,too-many-locals

    x, y, z = dataset
    x, y, z = shuffle(x, y, z)
    x, y, z = sort_by_label(x, y, z)

    if (x.shape[0] % num_partitions) and (not accept_imbalanced):
        raise ValueError(
            """Total number of samples must be a multiple of `num_partitions`.
               If imbalanced classes are allowed, set
               `accept_imbalanced=True`."""
        )

    num_samples = num_partitions * [0]
    for j in range(x.shape[0]):
        num_samples[j % num_partitions] += 1

    # Get number of classes and verify if they matching with
    classes, start_indices = np.unique(z, return_index=True)
    # Make sure that concentration is np.array and
    # check if concentration is appropriate
    concentration = np.asarray(concentration)

    # Check if concentration is Inf, if so create uniform partitions
    partitions: List[XYZ] = [(_, _, _) for _ in range(num_partitions)]
    if float("inf") in concentration:
        partitions = create_partitions(
            unpartitioned_dataset=(x, y, z),
            iid_fraction=1.0,
            num_partitions=num_partitions,
        )
        dirichlet_dist = get_partitions_distributions(partitions)[0]

        return partitions, dirichlet_dist

    if concentration.size == 1:
        concentration = np.repeat(concentration, classes.size)
    elif concentration.size != classes.size:  # Sequence
        raise ValueError(
            f"The size of the provided concentration ({concentration.size}) ",
            f"must be either 1 or equal number of classes {classes.size})",
        )

    # Split into list of list of samples per class
    list_samples_per_class: List[List[np.ndarray]] = split_array_at_indices(
        x, start_indices
    )
    list_sensitive_features_per_class: List[List[np.ndarray]] = [[]] * len(
        list_samples_per_class
    )
    for class_index, index_list in enumerate(list_samples_per_class):
        for index in list(index_list):
            list_sensitive_features_per_class[class_index].append(y[index])

    if dirichlet_dist is None:
        dirichlet_dist = np.random.default_rng(seed).dirichlet(
            alpha=concentration, size=num_partitions
        )

    if dirichlet_dist.size != 0:
        if dirichlet_dist.shape != (num_partitions, classes.size):
            raise ValueError(
                f"""The shape of the provided dirichlet distribution
                 ({dirichlet_dist.shape}) must match the provided number
                  of partitions and classes ({num_partitions},{classes.size})"""
            )

    # Assuming balanced distribution
    empty_classes = classes.size * [False]
    for partition_id in range(num_partitions):
        partitions[partition_id], empty_classes = sample_without_replacement(
            distribution=dirichlet_dist[partition_id].copy(),
            list_samples=list_samples_per_class,
            list_sensitive_features_per_class=list_sensitive_features_per_class,
            num_samples=num_samples[partition_id],
            empty_classes=empty_classes,
        )

    return partitions, dirichlet_dist


def create_sensitive_partition(
    dataset: XYZ,
    dirichlet_dist: Optional[np.ndarray] = None,
    num_partitions: int = 100,
    concentration: Union[float, np.ndarray, List[float]] = 100000,
    accept_imbalanced: bool = False,
    seed: Optional[Union[int, SeedSequence, BitGenerator, Generator]] = None,
) -> Tuple[XYZList, np.ndarray]:
    """Simple function for testing purposes. It creates a partitioning
    of the dataset based on the sensitive attribute.
    In particular, it creates `num_partitions` partitions, each one containing
    all the samples with the same sensitive attribute value.

    Args:
        dataset (_type_): dataset we want to split
        num_partitions (_type_): number of partitions we want to create

    Returns:
        _type_: _description_
    """
    # pylint: disable=too-many-arguments,too-many-locals
    x, y, z = dataset
    x, y, z = shuffle(x, y, z)
    x, y, z = sort_by_sensitive_value(x, y, z)

    if (x.shape[0] % num_partitions) and (not accept_imbalanced):
        raise ValueError(
            """Total number of samples must be a multiple of `num_partitions`.
               If imbalanced classes are allowed, set
               `accept_imbalanced=True`."""
        )

    num_samples = num_partitions * [0]
    for j in range(x.shape[0]):
        num_samples[j % num_partitions] += 1

    # Get number of sensitive values and verify if they matching with
    sensitive_values, start_indices = np.unique(y, return_index=True)
    # Make sure that concentration is np.array and
    # check if concentration is appropriate
    concentration = np.asarray(concentration)

    # Check if concentration is Inf, if so create uniform partitions
    partitions: List[XYZ] = [(_, _, _) for _ in range(num_partitions)]
    # TODO FIXME
    # if float("inf") in concentration:
    #     partitions = create_partitions(
    #         unpartitioned_dataset=(x, y, z),
    #         iid_fraction=1.0,
    #         num_partitions=num_partitions,
    #     )
    #     dirichlet_dist = get_partitions_distributions(partitions)[0]

    #     return partitions, dirichlet_dist

    if concentration.size == 1:
        concentration = np.repeat(concentration, sensitive_values.size)
    elif concentration.size != sensitive_values.size:  # Sequence
        raise ValueError(
            f"The size of the provided concentration ({concentration.size}) ",
            f"must be either 1 or equal number of classes {sensitive_values.size})",
        )

    # Split into list of list of samples per class
    list_samples_per_sensitive_feature: List[List[np.ndarray]] = split_array_at_indices(
        x,
        start_indices,
    )

    list_class_per_sensitive_feature: List[List[np.ndarray]] = [[]] * len(
        list_samples_per_sensitive_feature,
    )

    for sensitive_feature, index_list in enumerate(list_samples_per_sensitive_feature):
        for index in list(index_list):
            list_class_per_sensitive_feature[sensitive_feature].append(z[index])

    if dirichlet_dist is None:
        dirichlet_dist = np.random.default_rng(seed).dirichlet(
            alpha=concentration,
            size=num_partitions,
        )

    if dirichlet_dist.size != 0:
        if dirichlet_dist.shape != (num_partitions, sensitive_values.size):
            raise ValueError(
                f"""The shape of the provided dirichlet distribution
                 ({dirichlet_dist.shape}) must match the provided number
                  of partitions and classes ({num_partitions},{sensitive_values.size})"""
            )
    # Assuming balanced distribution
    empty_classes = sensitive_values.size * [False]
    for partition_id in range(num_partitions):
        partitions[partition_id], empty_classes = sample_without_replacement_sensitive(
            distribution=dirichlet_dist[partition_id].copy(),
            list_samples_per_sensitive_feature=list_samples_per_sensitive_feature,
            list_class_per_sensitive_feature=list_class_per_sensitive_feature,
            num_samples=num_samples[partition_id],
            empty_classes=empty_classes,
        )

    return partitions, dirichlet_dist


def create_unbalanced_partitions(dataset, num_partitions=2):
    x, y, z = dataset
    x, y, z = shuffle(x, y, z)
    x, y, z = sort_by_sensitive_value(x, y, z)

    sensitive_values, start_indices = np.unique(y, return_index=True)

    partitions: List[XYZ] = [(_, _, _) for _ in range(num_partitions)]

    # just a stupid example
    print(f"Assigning {start_indices[1]} samples with Male = {y[0]} to user 0")
    partitions[0] = (
        x[: start_indices[1]],
        y[: start_indices[1]],
        z[: start_indices[1]],
    )
    print(
        f"Assigning {len(y) - start_indices[1]} samples with Male = {y[start_indices[1]]} to user 1"
    )

    partitions[1] = (
        x[start_indices[1] :],
        y[start_indices[1] :],
        z[start_indices[1] :],
    )

    return partitions


def create_unbalanced_partitions_max_size(
    dataset, num_partitions, max_size, unbalanced_ratio
):
    x, y, z = dataset
    x, y, z = shuffle(x, y, z)
    x, y, z = sort_by_sensitive_value(x, y, z)
    if len(x) // num_partitions < max_size:
        raise ValueError("Invalid max_size")

    sensitive_values, start_indices = np.unique(y, return_index=True)
    start_indices = np.append(start_indices, len(y))

    groups_indexes = [
        y[start_indices[i] : start_indices[i + 1]] for i in range(len(sensitive_values))
    ]

    partitions: List[XYZ] = [(_, _, _) for _ in range(num_partitions)]

    for partition_id in range(num_partitions):
        underrepresented_group = np.random.choice(list(set(y)))
        # we want to sample unbalanced_ratio percentage from the underrepresented group
        # and 1 - unbalanced_ratio from the other groups:
        # unbalanced_ratio * num_samples = num_samples_underrepresented_group
        num_samples_underrepresented_group = int(unbalanced_ratio * max_size)
        num_samples_other_groups = max_size - num_samples_underrepresented_group

        # now sample num_samples_underrepresented_group from the underrepresented group
        # and num_samples_other_groups from the other groups
        index_underrepresented_group = np.where(y == underrepresented_group)[0]
        selected_samples_underrepresented = np.random.choice(
            index_underrepresented_group,
            num_samples_underrepresented_group,
            replace=False,
        )

        index_other_groups = np.where(y != underrepresented_group)[0]
        selected_samples_other_groups = np.random.choice(
            index_other_groups, num_samples_other_groups, replace=False
        )
        # print(len(selected_samples_other_groups), print(len(selected_samples_underrepresented)))
        partitions[partition_id] = (
            x[
                np.concatenate(
                    (selected_samples_underrepresented, selected_samples_other_groups)
                )
            ],
            y[
                np.concatenate(
                    (selected_samples_underrepresented, selected_samples_other_groups)
                )
            ],
            z[
                np.concatenate(
                    (selected_samples_underrepresented, selected_samples_other_groups)
                )
            ],
        )

        # remove the selected samples from the dataset
        x = np.delete(
            x,
            np.concatenate(
                (selected_samples_underrepresented, selected_samples_other_groups)
            ),
            axis=0,
        )
        y = np.delete(
            y,
            np.concatenate(
                (selected_samples_underrepresented, selected_samples_other_groups)
            ),
            axis=0,
        )
        z = np.delete(
            z,
            np.concatenate(
                (selected_samples_underrepresented, selected_samples_other_groups)
            ),
            axis=0,
        )

    return partitions


def create_single_unbalanced_partition_max_size(
    dataset, num_partitions, max_size, unbalanced_ratio
):
    x, y, z = dataset
    x, y, z = shuffle(x, y, z)
    x, y, z = sort_by_sensitive_value(x, y, z)
    if len(x) // num_partitions < max_size:
        raise ValueError("Invalid max_size")
    print(Counter(y))
    sensitive_values, start_indices = np.unique(y, return_index=True)
    start_indices = np.append(start_indices, len(y))

    groups_indexes = [
        y[start_indices[i] : start_indices[i + 1]] for i in range(len(sensitive_values))
    ]

    partitions: List[XYZ] = [(_, _, _) for _ in range(num_partitions)]

    # create the first partition [BALANCED]
    num_samples_underrepresented_group = max_size // 2
    num_samples_other_groups = max_size // 2
    underrepresented_group = 1
    # now sample num_samples_underrepresented_group from the underrepresented group
    # and num_samples_other_groups from the other groups
    index_underrepresented_group = np.where(y == underrepresented_group)[0]
    selected_samples_underrepresented = np.random.choice(
        index_underrepresented_group, num_samples_underrepresented_group, replace=False
    )

    index_other_groups = np.where(y != underrepresented_group)[0]
    selected_samples_other_groups = np.random.choice(
        index_other_groups, num_samples_other_groups, replace=False
    )
    # print(len(selected_samples_other_groups), print(len(selected_samples_underrepresented)))

    partitions[0] = (
        x[
            np.concatenate(
                (selected_samples_underrepresented, selected_samples_other_groups)
            )
        ],
        y[
            np.concatenate(
                (selected_samples_underrepresented, selected_samples_other_groups)
            )
        ],
        z[
            np.concatenate(
                (selected_samples_underrepresented, selected_samples_other_groups)
            )
        ],
    )

    # remove the selected samples from the dataset
    x = np.delete(
        x,
        np.concatenate(
            (selected_samples_underrepresented, selected_samples_other_groups)
        ),
        axis=0,
    )
    y = np.delete(
        y,
        np.concatenate(
            (selected_samples_underrepresented, selected_samples_other_groups)
        ),
        axis=0,
    )
    z = np.delete(
        z,
        np.concatenate(
            (selected_samples_underrepresented, selected_samples_other_groups)
        ),
        axis=0,
    )

    print(Counter(partitions[0][1]))

    # The second partition instead is unbalanced
    num_samples_underrepresented_group = int(unbalanced_ratio * max_size)
    num_samples_other_groups = max_size - num_samples_underrepresented_group
    print(num_samples_underrepresented_group, num_samples_other_groups)
    # now sample num_samples_underrepresented_group from the underrepresented group
    # and num_samples_other_groups from the other groups
    index_underrepresented_group = np.where(y == underrepresented_group)[0]
    selected_samples_underrepresented = np.random.choice(
        index_underrepresented_group, num_samples_underrepresented_group, replace=False
    )

    index_other_groups = np.where(y != underrepresented_group)[0]
    selected_samples_other_groups = np.random.choice(
        index_other_groups, num_samples_other_groups, replace=False
    )
    # print(len(selected_samples_other_groups), print(len(selected_samples_underrepresented)))

    partitions[1] = (
        x[
            np.concatenate(
                (selected_samples_underrepresented, selected_samples_other_groups)
            )
        ],
        y[
            np.concatenate(
                (selected_samples_underrepresented, selected_samples_other_groups)
            )
        ],
        z[
            np.concatenate(
                (selected_samples_underrepresented, selected_samples_other_groups)
            )
        ],
    )

    # remove the selected samples from the dataset
    x = np.delete(
        x,
        np.concatenate(
            (selected_samples_underrepresented, selected_samples_other_groups)
        ),
        axis=0,
    )
    y = np.delete(
        y,
        np.concatenate(
            (selected_samples_underrepresented, selected_samples_other_groups)
        ),
        axis=0,
    )
    z = np.delete(
        z,
        np.concatenate(
            (selected_samples_underrepresented, selected_samples_other_groups)
        ),
        axis=0,
    )

    return partitions


# def create_reduced_partitions_max_size(dataset, num_partitions, max_size):
#     x, y, z = dataset
#     x, y, z = shuffle(x, y, z)
#     x, y, z = sort_by_sensitive_value(x, y, z)


#     sensitive_values, start_indices = np.unique(y, return_index=True)

#     partitions: List[XYZ] = [(_, _, _) for _ in range(num_partitions)]

#     # just a stupid example
#     print(f"Assigning {start_indices[1]} samples with Male = {y[0]} to user 0")
#     partitions[0] = (
#         x[: start_indices[1]],
#         y[: start_indices[1]],
#         z[: start_indices[1]],
#     )
#     print(f"Assigning {len(y) - start_indices[1]} samples with Male = {y[start_indices[1]]} to user 1")

#     partitions[1] = (
#         x[start_indices[1] :],
#         y[start_indices[1] :],
#         z[start_indices[1] :],
#     )

#     return partitions


def partition_10_nodes(dataset, num_partitions, max_size, num_underrepresented_nodes=1):
    x, y, z = dataset
    x, y, z = shuffle(x, y, z)
    x, y, z = sort_by_sensitive_value(x, y, z)

    sensitive_values, start_indices = np.unique(y, return_index=True)

    partitions: List[XYZ] = [(_, _, _) for _ in range(num_partitions)]

    represented_nodes = num_partitions - num_underrepresented_nodes
    print(len(x))
    print(sensitive_values)
    print(start_indices)
    size_represented = start_indices[1] // 100
    size_underrepresented = 2000 // 100
    group_represented = [
        [i * size_represented, i * size_represented + size_represented]
        for i in range(num_partitions)
    ]
    group_represented[-1] = [
        group_represented[-1][0],
        group_represented[-1][0] + size_underrepresented,
    ]

    size_non_represented = size_underrepresented
    group_non_represented = [
        [
            start_indices[1] + (i * size_non_represented),
            start_indices[1] + (i * size_non_represented + size_non_represented),
        ]
        for i in range(num_partitions)
    ]
    group_non_represented[-1] = [
        group_non_represented[-1][0],
        group_non_represented[-1][0] + size_represented,
    ]

    print(type(x[group_represented[0][0] : group_represented[0][1]]))
    for node_id in range(num_partitions):
        partitions[node_id] = (
            np.concatenate(
                (
                    x[group_represented[node_id][0] : group_represented[node_id][1]],
                    x[
                        group_non_represented[node_id][0] : group_non_represented[
                            node_id
                        ][1]
                    ],
                )
            ),
            np.concatenate(
                (
                    y[group_represented[node_id][0] : group_represented[node_id][1]],
                    y[
                        group_non_represented[node_id][0] : group_non_represented[
                            node_id
                        ][1]
                    ],
                )
            ),
            np.concatenate(
                (
                    z[group_represented[node_id][0] : group_represented[node_id][1]],
                    z[
                        group_non_represented[node_id][0] : group_non_represented[
                            node_id
                        ][1]
                    ],
                )
            ),
        )
        print(Counter(partitions[node_id][1]))
    return partitions
