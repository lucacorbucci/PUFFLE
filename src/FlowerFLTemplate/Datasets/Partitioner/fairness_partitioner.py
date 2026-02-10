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
        samples_per_client: int | None = None,
        distribution_mode: str = "per_group",
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
        self._samples_per_client = samples_per_client
        self._distribution_mode = distribution_mode
        self._partitions: dict[int, Dataset] = {}
        self.client_types: dict[int, str] = {}  # "fair" or "unfair"
        self._discarded_samples_count: int = 0

        # Validate distribution_mode
        if self._distribution_mode not in ["per_group", "representative"]:
            msg = (
                f"distribution_mode must be 'per_group' or 'representative', "
                f"got '{self._distribution_mode}'"
            )
            raise ValueError(msg)

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

    def _representative_distribution(self) -> tuple[list[pd.DataFrame], dict[tuple[Any, Any], pd.DataFrame]]:
        """
        Implements representative diversity approach (OLD behavior).

        Shuffles the entire dataset and sequentially allocates samples to clients.
        This creates natural variance between clients.

        Returns:
            tuple: (nodes, remaining_data_by_group)
                - nodes: List of DataFrames, one per client
                - remaining_data_by_group: Dict mapping (target, sensitive) -> DataFrame of remaining samples

        """
        df = self._dataset.to_pandas()

        # Shuffle with seed for reproducibility
        df_shuffled = df.sample(frac=1, random_state=self._seed).reset_index(drop=True)

        # Determine samples per client
        if self._samples_per_client:
            samples_per_node = self._samples_per_client
            # Validate total requested doesn't exceed available
            total_requested = samples_per_node * self._num_partitions
            if total_requested > len(df_shuffled):
                msg = (
                    f"Requested {total_requested} total samples "
                    f"({samples_per_node} per client x {self._num_partitions} clients) "
                    f"but only {len(df_shuffled)} samples available in dataset."
                )
                raise ValueError(msg)
        else:
            samples_per_node = len(df_shuffled) // self._num_partitions

        # Sequential allocation
        nodes = []
        for i in range(self._num_partitions):
            start = i * samples_per_node
            end = start + samples_per_node
            node_df = df_shuffled.iloc[start:end].copy()
            nodes.append(node_df)

        # Store remaining samples grouped by (target, sensitive)
        remaining_start = self._num_partitions * samples_per_node
        remaining_df = df_shuffled.iloc[remaining_start:].copy()

        # Initialize with all possible groups
        remaining_data_by_group = {
            (t, s): []
            for t in [0, 1]
            for s in [0, 1]
        }

        if len(remaining_df) > 0:
            # Group remaining samples
            remaining_df["_group_temp"] = list(
                zip(
                    remaining_df[self._target_attribute],
                    remaining_df[self._sensitive_attribute],
                    strict=False,
                )
            )
            for group in remaining_df["_group_temp"].unique():
                group_df = remaining_df[remaining_df["_group_temp"] == group].copy()
                group_df = group_df.drop(columns=["_group_temp"])
                remaining_data_by_group[group] = group_df

        return nodes, remaining_data_by_group

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

        # Set seeds
        np.random.seed(self._seed)
        random.seed(self._seed)

        # Define unique groups (needed by both modes)
        groups = list(
            zip(df[self._target_attribute], df[self._sensitive_attribute], strict=False)
        )
        unique_groups = sorted(set(groups))  # sort for determinism

        # Choose distribution strategy based on mode
        if self._distribution_mode == "representative":
            # Use OLD representative diversity approach (random sampling)
            nodes, remaining_data_by_group = self._representative_distribution()

            # Continue with unfair node creation using OLD swap-based approach
            # This maintains samples_per_client by swapping samples between fair and unfair nodes
            number_unfair_nodes = int(self._num_partitions * self._ratio_unfair_clients)
            number_fair_nodes = self._num_partitions - number_unfair_nodes

            fair_nodes = nodes[:number_fair_nodes]
            nodes_to_unfair = nodes[number_fair_nodes:]

            # Record types
            for i in range(number_fair_nodes):
                self.client_types[i] = "fair"
            for i in range(number_unfair_nodes):
                self.client_types[number_fair_nodes + i] = "unfair"

            # OLD-style unfair node creation (swap-based to maintain sample counts)
            group_to_reduce = self._group_to_reduce
            group_to_increment = self._group_to_increment

            removed_samples = []
            number_of_samples_to_add_per_node = []
            processed_unfair_nodes = []

            # Step 1: Remove samples from unfair nodes (group_to_reduce)
            for node_df in nodes_to_unfair:
                mask = (node_df[self._target_attribute] == group_to_reduce[0]) & (
                    node_df[self._sensitive_attribute] == group_to_reduce[1]
                )
                count_sensitive = mask.sum()

                # Determine how many to remove
                ratio = np.random.uniform(
                    self._ratio_unfairness[0], self._ratio_unfairness[1]
                )
                samples_to_remove_count = int(count_sensitive * ratio)

                # Remove samples
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

            # Step 2: Distribute removed samples to fair nodes
            all_removed = pd.concat(removed_samples, ignore_index=True) if removed_samples else pd.DataFrame()

            if len(all_removed) > 0 and number_fair_nodes > 0:
                samples_per_fair_node = len(all_removed) // number_fair_nodes
                current_idx = 0

                for i in range(number_fair_nodes):
                    end_idx = current_idx + samples_per_fair_node
                    if i == number_fair_nodes - 1:  # Last node gets remainder
                        to_add = all_removed.iloc[current_idx:]
                    else:
                        to_add = all_removed.iloc[current_idx:end_idx]

                    if len(to_add) > 0:
                        fair_nodes[i] = pd.concat([fair_nodes[i], to_add], ignore_index=True)
                    current_idx = end_idx

            # Step 3: Remove samples from fair nodes (group_to_increment) to give to unfair nodes
            if group_to_increment and number_fair_nodes > 0:
                total_samples_needed = sum(number_of_samples_to_add_per_node)
                samples_to_remove_per_fair_node = total_samples_needed // number_fair_nodes

                samples_for_unfair = []

                for idx, fair_node_df in enumerate(fair_nodes):
                    mask = (fair_node_df[self._target_attribute] == group_to_increment[0]) & (
                        fair_node_df[self._sensitive_attribute] == group_to_increment[1]
                    )

                    # Remove samples from fair node
                    available = fair_node_df[mask]
                    to_remove_count = min(samples_to_remove_per_fair_node, len(available))

                    if to_remove_count > 0:
                        indices_to_remove = available.sample(
                            n=to_remove_count, random_state=self._seed
                        ).index

                        removed_from_fair = fair_node_df.loc[indices_to_remove]
                        samples_for_unfair.append(removed_from_fair)

                        # Update fair node (remove these samples)
                        fair_nodes[idx] = fair_node_df.drop(indices_to_remove)

                # Step 4: Add samples to unfair nodes
                all_for_unfair = pd.concat(samples_for_unfair, ignore_index=True) if samples_for_unfair else pd.DataFrame()

                if len(all_for_unfair) > 0:
                    current_idx = 0
                    for i, needed in enumerate(number_of_samples_to_add_per_node):
                        end_idx = min(current_idx + needed, len(all_for_unfair))
                        to_add = all_for_unfair.iloc[current_idx:end_idx]

                        if len(to_add) > 0:
                            processed_unfair_nodes[i] = pd.concat(
                                [processed_unfair_nodes[i], to_add], ignore_index=True
                            )
                        current_idx = end_idx

            # Combine fair and unfair nodes
            all_nodes = fair_nodes + processed_unfair_nodes

            # Store in partitions
            for i, node_df in enumerate(all_nodes):
                # Remove temporary column if it exists
                if "_group_temp" in node_df.columns:
                    node_df = node_df.drop(columns=["_group_temp"])
                self._partitions[i] = Dataset.from_pandas(node_df)

            self._discarded_samples_count = 0  # Representative mode doesn't discard in the old way
            return  # Representative mode is complete


        # "per_group" mode - Balanced fair clients, consistent unfair clients
        
        # Add temporary column for group id
        df["_group_temp"] = groups

        # Split data by group
        data_by_group = {g: df[df["_group_temp"] == g].copy() for g in unique_groups}

        # Calculate client counts
        number_unfair_nodes = int(self._num_partitions * self._ratio_unfair_clients)
        number_fair_nodes = self._num_partitions - number_unfair_nodes

        # Record client types
        for i in range(number_fair_nodes):
            self.client_types[i] = "fair"
        for i in range(number_unfair_nodes):
            self.client_types[number_fair_nodes + i] = "unfair"

        # Determine samples per client
        if self._samples_per_client:
            samples_per_client = self._samples_per_client
            # Validate total requested doesn't exceed available
            total_requested = samples_per_client * self._num_partitions
            if total_requested > len(df):
                msg = (
                    f"Requested {total_requested} total samples "
                    f"({samples_per_client} per client x {self._num_partitions} clients) "
                    f"but only {len(df)} samples available in dataset."
                )
                raise ValueError(msg)
        else:
            samples_per_client = len(df) // self._num_partitions

        # Calculate base samples per group (for balanced distribution)
        samples_per_group_base = samples_per_client // 4
        remainder = samples_per_client % 4

        # === FAIR CLIENTS: Balanced distribution with small variance ===
        fair_nodes = []
        
        for i in range(number_fair_nodes):
            # Add small random variance (±2 samples per group) while maintaining balance
            variance = np.random.randint(-2, 3, size=4, dtype=np.int32)
            # Ensure sum is 0 to maintain total sample count
            variance = variance - int(variance.mean())
            
            node_parts = []
            for idx, group in enumerate(unique_groups):
                # Calculate samples for this group
                n_samples = samples_per_group_base + int(variance[idx])
                
                # Distribute remainder to first groups
                if idx < remainder:
                    n_samples += 1
                
                # Ensure we don't request more samples than available
                available = len(data_by_group[group])
                if n_samples > available:
                    n_samples = available
                
                # Sample from this group
                if n_samples > 0:
                    group_samples = data_by_group[group].sample(
                        n=n_samples,
                        random_state=self._seed + i * 4 + idx,
                        replace=False
                    )
                    node_parts.append(group_samples)
                    # Remove sampled data to avoid reuse
                    data_by_group[group] = data_by_group[group].drop(group_samples.index)
            
            # Combine all groups for this client
            if node_parts:
                fair_node = pd.concat(node_parts, ignore_index=True)
                fair_nodes.append(fair_node)
            else:
                fair_nodes.append(pd.DataFrame())

        # === UNFAIR CLIENTS: Consistent unfairness with small variance ===
        
        # Calculate base reduction ratio (same for all unfair clients)
        base_reduction_ratio = np.random.uniform(
            self._ratio_unfairness[0],
            self._ratio_unfairness[1]
        )
        
        unfair_nodes = []
        
        for i in range(number_unfair_nodes):
            # Add small variance to reduction ratio (±5%)
            reduction_ratio = base_reduction_ratio + np.random.uniform(-0.05, 0.05)
            # Clip to ensure it stays within bounds
            reduction_ratio = np.clip(
                reduction_ratio,
                self._ratio_unfairness[0],
                self._ratio_unfairness[1]
            )
            
            # Calculate samples for each group
            samples_reduce = int(samples_per_group_base * (1 - reduction_ratio))
            samples_increment = int(samples_per_group_base * (1 + reduction_ratio))
            samples_other = samples_per_group_base
            
            # Adjust to maintain total sample count
            total = samples_reduce + samples_increment + 2 * samples_other
            adjustment = samples_per_client - total
            
            # Distribute adjustment to "other" groups
            samples_other_1 = samples_other + adjustment // 2
            samples_other_2 = samples_other + (adjustment - adjustment // 2)
            
            node_parts = []
            other_idx = 0
            
            for group in unique_groups:
                if group == self._group_to_reduce:
                    n_samples = samples_reduce
                elif group == self._group_to_increment:
                    n_samples = samples_increment
                else:
                    # Distribute to other groups
                    n_samples = samples_other_1 if other_idx == 0 else samples_other_2
                    other_idx += 1
                
                # Ensure we don't request more samples than available
                available = len(data_by_group[group])
                if n_samples > available:
                    n_samples = available
                
                # Sample from this group
                if n_samples > 0:
                    group_samples = data_by_group[group].sample(
                        n=n_samples,
                        random_state=self._seed + number_fair_nodes + i * 4 + unique_groups.index(group),
                        replace=False
                    )
                    node_parts.append(group_samples)
                    # Remove sampled data to avoid reuse
                    data_by_group[group] = data_by_group[group].drop(group_samples.index)
            
            # Combine all groups for this client
            if node_parts:
                unfair_node = pd.concat(node_parts, ignore_index=True)
                unfair_nodes.append(unfair_node)
            else:
                unfair_nodes.append(pd.DataFrame())

        # Calculate discarded samples
        total_remaining = sum(len(data_by_group[g]) for g in unique_groups)
        self._discarded_samples_count = total_remaining

        # Cleanup: Remove temporary column
        for i in range(len(fair_nodes)):
            if "_group_temp" in fair_nodes[i].columns:
                fair_nodes[i] = fair_nodes[i].drop(columns=["_group_temp"])

        for i in range(len(unfair_nodes)):
            if "_group_temp" in unfair_nodes[i].columns:
                unfair_nodes[i] = unfair_nodes[i].drop(columns=["_group_temp"])

        # Store partitions - Convert back to HF Dataset
        all_nodes = fair_nodes + unfair_nodes
        
        for i, df_part in enumerate(all_nodes):
            # Ensure index is reset
            df_part = df_part.reset_index(drop=True)
            self._partitions[i] = Dataset.from_pandas(df_part)

