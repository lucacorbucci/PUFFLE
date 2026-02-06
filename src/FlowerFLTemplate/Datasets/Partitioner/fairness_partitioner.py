import random
from typing import Any

import numpy as np
import pandas as pd
from datasets import Dataset
from flwr_datasets.partitioner.partitioner import Partitioner


class FairnessPartitioner(Partitioner):
    """
    Partitioner that creates specific unfair/fair data distributions across clients.

    It divides the data such that some clients are "unfair" (biased against a specific group)
    and some are "fair" (balanced). It ensures clients have equal sample sizes (if possible)
    and moves data between clients to achieve the desired bias.
    """

    def __init__(
        self,
        num_partitions: int,
        sensitive_attribute: str,
        target_attribute: str,
        ratio_unfair_clients: float,
        group_to_reduce: tuple[Any, Any],
        ratio_unfairness: tuple[float, float],
        group_to_increment: tuple[Any, Any] | None = None,
        seed: int = 42,
        dataset: Dataset | None = None,
    ) -> None:
        super().__init__()
        self._num_partitions = num_partitions
        self._sensitive_attribute = sensitive_attribute
        self._target_attribute = target_attribute
        self._ratio_unfair_clients = ratio_unfair_clients
        self._group_to_reduce = group_to_reduce
        self._ratio_unfairness = ratio_unfairness
        self._group_to_increment = group_to_increment
        self._seed = seed
        self._dataset = dataset
        self._partitions: dict[int, Dataset] = {}
        self.client_types: dict[int, str] = {}  # "fair" or "unfair"
        self._discarded_samples_count: int = 0

        # Determine group to increment if not provided (assume binary opposite)
        # This logic might need refinement if not binary, but for now we assume binary or user provided.
        # Actually user should provide it or we infer it.
        # For now, if not provided, we might fail or try to infer later.

        if self._dataset is not None:
            self._partition_data()

    @property
    def dataset(self) -> Dataset:
        """Get the dataset."""
        return self._dataset

    @dataset.setter
    def dataset(self, value: Dataset) -> None:
        """Set the dataset and re-partition."""
        self._dataset = value
        self._partition_data()

    @property
    def discarded_samples_count(self) -> int:
        """Return the number of samples that were discarded during partitioning."""
        return self._discarded_samples_count

    @property
    def num_partitions(self) -> int:
        """Return the number of partitions."""
        return self._num_partitions

    def load_partition(self, partition_id: int) -> Dataset:
        """Load a single partition."""
        if partition_id not in self._partitions:
            msg = f"Partition {partition_id} not found. Ensure dataset is set and partition_id is valid."
            raise ValueError(msg)
        return self._partitions[partition_id]

    def _validate_binary_attributes(self, df: pd.DataFrame) -> None:
        """Validate that sensitive and target attributes are binary (only contain 0 and 1)."""
        for attr_name, attr_col in [
            ("sensitive_attribute", self._sensitive_attribute),
            ("target_attribute", self._target_attribute),
        ]:
            unique_values = sorted(df[attr_col].unique())
            if not set(unique_values).issubset({0, 1}):
                msg = (
                    f"{attr_name} '{attr_col}' must be binary (contain only 0 and 1). "
                    f"Found values: {unique_values}. "
                    f"Please preprocess your dataset to binarize this attribute."
                )
                raise ValueError(msg)

    def _partition_data(self) -> None:
        """
        Performs the data partitioning logic.
        """
        if self._dataset is None:
            return

        # Convert HF Dataset to Pandas for manipulation
        df = self._dataset.to_pandas()

        # Validate that attributes are binary
        self._validate_binary_attributes(df)

        # Extract X, y, z
        # We perform operations on the dataframe directly to keep all columns

        # Set seeds
        np.random.seed(self._seed)
        random.seed(self._seed)

        # 1. Egalitarian Split (Initial balanced split)
        # 2. Create Unfair Nodes

        # For compatibility with legacy logic, we'll treat rows as dicts
        # or work with DataFrame. Let's work with DataFrame and convert logic.

        # Logic adapted from tabular_data_loader.py

        # 1. Egalitarian approach: Distribute data such that each node has same amount of data and same ratio of groups
        # We identify unique groups based on (target, sensitive)

        # Logic adapted to allow unfairness (natural distribution) in Fair nodes
        # Instead of finding global min, we find min per group to distribute equally across nodes.

        groups = list(
            zip(df[self._target_attribute], df[self._sensitive_attribute], strict=False)
        )
        # Add a temporary column for group id
        df["_group_temp"] = groups
        unique_groups = sorted(set(groups))  # sort for determinism

        # Split data by group
        data_by_group = {g: df[df["_group_temp"] == g].copy() for g in unique_groups}

        # Calculate samples per node PER GROUP.
        # This allows "Fair" nodes to have the natural imbalance of the dataset,
        # rather than forcing them to be perfectly balanced (which discards data).

        max_ratio = self._ratio_unfairness[1]
        number_unfair_nodes = int(self._num_partitions * self._ratio_unfair_clients)

        samples_per_group_per_node = {}  # Dict[Group, int]

        for g in unique_groups:
            total_count = len(data_by_group[g])
            if self._group_to_increment and g == self._group_to_increment:
                # This group needs reserve for Unfair nodes to be incremented later
                denominator = self._num_partitions + number_unfair_nodes * max_ratio
                base = total_count // denominator
            else:
                # Standard distribution
                base = total_count // self._num_partitions

            samples_per_group_per_node[g] = int(base)

        # Prepare lists for nodes
        node_dfs = [[] for _ in range(self._num_partitions)]

        # Distribute equally (per group)
        remaining_data_by_group = {g: [] for g in unique_groups}
        # Also track discarded/remaining samples that didn't fit into the equal split or reserved pool

        # In this new logic, "remaining" from the equal split IS what we might call "discarded"
        # UNLESS it is used for the "increment" reserve.

        # Let's clarify:
        # For group_to_increment:
        #   We use `base` for all nodes.
        #   We reserve `base * number_unfair_nodes * max_ratio` roughly for incrementing.
        #   Whatever is left after (num_partitions * base + reserve) is truly discarded.
        # For other groups:
        #   We use `base` for all nodes.
        #   Whatever is left after (num_partitions * base) is truly discarded.

        for g in unique_groups:
            group_df = (
                data_by_group[g]
                .sample(frac=1, random_state=self._seed)
                .reset_index(drop=True)
            )
            count_per_node = samples_per_group_per_node[g]

            for i in range(self._num_partitions):
                start = i * count_per_node
                end = (i + 1) * count_per_node
                subset = group_df.iloc[start:end]
                node_dfs[i].append(subset)

            used_count = self._num_partitions * count_per_node

            if len(group_df) > used_count:
                remaining_df = group_df.iloc[used_count:]
                remaining_data_by_group[g] = remaining_df

        # Concatenate parts for each node (initial fair nodes)
        nodes = [pd.concat(parts, ignore_index=True) for parts in node_dfs]

        number_fair_nodes = self._num_partitions - number_unfair_nodes

        fair_nodes = nodes[:number_fair_nodes]
        nodes_to_unfair = nodes[number_fair_nodes:]

        # Record types
        for i in range(number_fair_nodes):
            self.client_types[i] = "fair"
        for i in range(number_unfair_nodes):
            self.client_types[number_fair_nodes + i] = "unfair"

        # 2. Create Unfair Nodes logic

        group_to_increment = self._group_to_increment
        group_to_reduce = self._group_to_reduce

        # We will track samples that are safe to redistribute (random remainders)
        # and samples that must be discarded to avoid sign-flipping (toxic waste).
        redistribution_pool_dfs = []

        for g in unique_groups:
            if g == group_to_increment:
                # Handled below
                pass
            elif isinstance(remaining_data_by_group[g], pd.DataFrame):
                # Clean remainder
                redistribution_pool_dfs.append(remaining_data_by_group[g])
                remaining_data_by_group[g] = []

        removed_samples = []
        number_of_samples_to_add_per_node = []

        processed_unfair_nodes = []

        for node_df in nodes_to_unfair:
            # Count target samples (to reduce)
            mask = (node_df[self._target_attribute] == group_to_reduce[0]) & (
                node_df[self._sensitive_attribute] == group_to_reduce[1]
            )
            count_sensitive = mask.sum()

            # Determine how many to remove
            ratio = np.random.uniform(
                self._ratio_unfairness[0], self._ratio_unfairness[1]
            )
            samples_to_remove_count = int(count_sensitive * ratio)

            # Identify indices to remove
            indices_to_remove = (
                node_df[mask]
                .sample(n=samples_to_remove_count, random_state=self._seed)
                .index
            )

            removed = node_df.loc[indices_to_remove]
            removed_samples.append(removed)

            kept = node_df.drop(indices_to_remove)
            processed_unfair_nodes.append(kept)

            number_of_samples_to_add_per_node.append(samples_to_remove_count)

        # Toxic waste: explicitly removed samples.
        # These will be discarded to maintain consistent disparity direction.
        toxic_waste_count = sum(len(df) for df in removed_samples)

        # Now increment group_to_increment in Unfair nodes using RESERVED data

        if group_to_increment:
            # Prepare the pool from remaining data
            if isinstance(
                remaining_data_by_group.get(group_to_increment), pd.DataFrame
            ):
                pool_increment = remaining_data_by_group[group_to_increment]
            else:
                pool_increment = pd.DataFrame(columns=df.columns)

            # Distribute this pool to unfair nodes
            current_pool_idx = 0
            for i, node in enumerate(processed_unfair_nodes):
                needed = number_of_samples_to_add_per_node[i]

                if current_pool_idx + needed <= len(pool_increment):
                    to_add = pool_increment.iloc[
                        current_pool_idx : current_pool_idx + needed
                    ]
                    processed_unfair_nodes[i] = pd.concat(
                        [node, to_add], ignore_index=True
                    )
                    current_pool_idx += needed
                else:
                    # Take what's left
                    left = len(pool_increment) - current_pool_idx
                    if left > 0:
                        to_add = pool_increment.iloc[current_pool_idx:]
                        processed_unfair_nodes[i] = pd.concat(
                            [node, to_add], ignore_index=True
                        )
                        current_pool_idx += left

            # Anything left in pool_increment after filling unfair nodes is clean waste
            if current_pool_idx < len(pool_increment):
                unused_reserve = pool_increment.iloc[current_pool_idx:]
                redistribution_pool_dfs.append(unused_reserve)
        # Redistribute "Clean" waste to Fair nodes if any exist
        if redistribution_pool_dfs:
            total_clean_df = pd.concat(redistribution_pool_dfs, ignore_index=True)

            if number_fair_nodes > 0:
                # Shuffle for randomness
                total_clean_df = total_clean_df.sample(
                    frac=1, random_state=self._seed
                ).reset_index(drop=True)

                # We distribute as evenly as possible
                # Split by index to maintain DataFrame type
                index_chunks = np.array_split(total_clean_df.index, number_fair_nodes)

                for i in range(number_fair_nodes):
                    if i < len(index_chunks):
                        chunk_indices = index_chunks[i]
                        if len(chunk_indices) > 0:
                            to_add = total_clean_df.loc[chunk_indices]
                            fair_nodes[i] = pd.concat(
                                [fair_nodes[i], to_add], ignore_index=True
                            )
                # Discarded is just the toxic part
                self._discarded_samples_count = toxic_waste_count
            else:
                # Truly discarded = toxic + clean
                self._discarded_samples_count = toxic_waste_count + len(total_clean_df)
        else:
            self._discarded_samples_count = toxic_waste_count

        # Cleanup
        # Remove temporary column
        for i in range(len(fair_nodes)):
            if "_group_temp" in fair_nodes[i].columns:
                fair_nodes[i] = fair_nodes[i].drop(columns=["_group_temp"])

        for i in range(len(processed_unfair_nodes)):
            if "_group_temp" in processed_unfair_nodes[i].columns:
                processed_unfair_nodes[i] = processed_unfair_nodes[i].drop(
                    columns=["_group_temp"]
                )

        # Store partitions
        # Convert back to HF Dataset

        for i, df_part in enumerate(fair_nodes + processed_unfair_nodes):
            # Ensure index is reset
            df_part = df_part.reset_index(drop=True)
            self._partitions[i] = Dataset.from_pandas(df_part)
