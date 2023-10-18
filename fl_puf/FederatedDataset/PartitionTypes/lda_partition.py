import numpy as np
import torch

from fl_puf.FederatedDataset.Utils.lda import create_lda_partitions


class LDAPartition:
    def do_partitioning(
        dataset: torch.utils.data.Dataset,
        num_partitions: int,
        alpha: float,
    ) -> dict:
        labels = np.array(dataset.targets)
        num_labels = len(np.unique(labels))
        idx = np.array(range(len(labels)))

        print(type(labels))
        print(type(idx))
        print(type(labels[0]))
        print(type(idx[0]))
        partitions, dirichlet_dist = create_lda_partitions(
            dataset=[idx, labels],
            num_partitions=num_partitions,
            concentration=alpha,
            accept_imbalanced=True,
            seed=42,
        )
        for partition in partitions:
            partition_zero = partition[1]
            hist, _ = np.histogram(partition_zero, bins=list(range(num_labels + 1)))
            print(f"Class histogram (alpha={alpha}, {num_labels} classes): {hist}")

        splitted_indexes_test = {
            str(index): partition[0] for index, partition in enumerate(partitions)
        }

        labels_per_cluster_test = {
            str(index): labels[partition[0]]
            for index, partition in enumerate(partitions)
        }
        return splitted_indexes_test, labels_per_cluster_test
