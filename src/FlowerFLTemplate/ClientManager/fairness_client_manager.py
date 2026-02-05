import random
from logging import INFO

import dill
from flwr.common import GetPropertiesIns
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from FlowerFLTemplate.ClientManager.client_manager import SimpleClientManager
from FlowerFLTemplate.Utils.preferences import Preferences


class FairnessClientManager(SimpleClientManager):
    """
    ClientManager that fetches partition IDs via get_properties and uses them
    for CID assignment. Samples EQUAL numbers of fair and unfair clients per round
    for training, validation, and test sets.
    """

    def __init__(self, preferences: Preferences, client_types: dict[int, str]) -> None:
        """
        Args:
            preferences: Configuration preferences.
            client_types: Dictionary mapping partition ID (int) to type ("fair" or "unfair").

        """
        super().__init__(preferences)
        self.client_types = client_types
        # Mapping from node_id (original CID) to partition_id
        self._node_to_partition: dict[str, int] = {}

    def register(self, client: ClientProxy) -> bool:
        """
        Registers a Flower ClientProxy instance with the manager.

        Fetches the partition_id from the client via get_properties and uses it as the CID.
        This enables proper mapping for fairness-based stratified sampling.

        Args:
            client (ClientProxy): The client to register.

        Returns:
            bool: True if registration succeeded, False if already registered.

        """
        # Check if already registered (by original node_id/cid)
        if client.cid in self.clients:
            return False

        try:
            # Fetch partition_id from client via get_properties
            # This should be instant now that FlowerClient uses lazy loading
            while True:
                res = client.get_properties(
                    ins=GetPropertiesIns(config={}), timeout=30.0, group_id=None
                )
                partition_id = res.properties.get("partition_id")
                if partition_id is not None:
                    break

            if partition_id is not None:
                # Store the mapping from original CID to partition_id
                original_cid = client.cid
                self._node_to_partition[original_cid] = int(partition_id)

                # Update client's CID to the partition_id
                client.cid = str(partition_id)
                log(
                    INFO,
                    f"Registered client: node_id={original_cid} -> partition_id={partition_id}",
                )
            else:
                log(
                    INFO,
                    f"Client {client.cid} did not return partition_id, keeping original CID",
                )
        except (TimeoutError, ValueError, KeyError, RuntimeError) as e:
            log(INFO, f"Failed to get properties for client {client.cid}: {e}")
            # Fall back to using original CID

        # Check again with updated CID
        if client.cid in self.clients:
            log(INFO, f"Client with CID {client.cid} already registered, skipping")
            return False

        self.clients[client.cid] = client
        self.clients_list.append(client.cid)

        if self.preferences.num_clients == len(self.clients_list):
            self._setup_simulation_sets()

            with self._cv:
                self._cv.notify_all()

        return True

    def _assign_unique_cid(self, client: ClientProxy) -> None:
        """
        Override to do nothing - we assign CIDs based on partition_id in register().
        """

    def _stratify_clients(self, client_list: list[str]) -> tuple[list[str], list[str]]:
        """
        Stratify clients into fair and unfair groups.

        Args:
            client_list: List of client CIDs (which are now partition IDs).

        Returns:
            tuple[list[str], list[str]]: (fair_clients, unfair_clients) sorted lists.

        """
        fair_clients = []
        unfair_clients = []

        for cid in client_list:
            try:
                cid_int = int(cid)
                c_type = self.client_types.get(cid_int, "unknown")
                if c_type == "fair":
                    fair_clients.append(cid)
                elif c_type == "unfair":
                    unfair_clients.append(cid)
                else:
                    # Default unknown to fair
                    fair_clients.append(cid)
            except ValueError:
                # String CID not convertible to int
                fair_clients.append(cid)

        # Sort for determinism
        fair_clients.sort(key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
        unfair_clients.sort(key=lambda x: (0, int(x)) if x.isdigit() else (1, x))

        return fair_clients, unfair_clients

    def _split_clients_equal(
        self, n_total: int
    ) -> tuple[list[str], list[str], list[str] | None]:
        """
        Split all clients into train, test, and optionally validation sets with equal
        fair/unfair distribution in each set.

        Args:
            n_total: Total number of clients (for reference).

        Returns:
            tuple: (train_clients, test_clients, validation_clients).

        """
        # First stratify all clients
        fair_clients, unfair_clients = self._stratify_clients(self.clients_list)

        log(INFO, f"Total fair clients: {len(fair_clients)}")
        log(INFO, f"Total unfair clients: {len(unfair_clients)}")

        # Calculate how many fair/unfair clients to allocate to each set
        # Test set: equal fair/unfair
        n_test = self.preferences.num_test_nodes
        n_test_per_type = n_test // 2  # Equal split

        test_fair = fair_clients[:n_test_per_type]
        test_unfair = unfair_clients[:n_test_per_type]
        test_clients = test_fair + test_unfair

        # Remaining clients after test
        remaining_fair = fair_clients[n_test_per_type:]
        remaining_unfair = unfair_clients[n_test_per_type:]

        # shuffle both remaining_fair and remaining_unfair
        # using node_shuffle_seed as seed. This is necessary
        # because we want to make sure that the validation set of clients
        # is changed every time we run a sweep to avoid overfitting.
        random.seed(self.preferences.node_shuffle_seed)
        random.shuffle(remaining_fair)
        random.shuffle(remaining_unfair)
        random.seed(self.preferences.seed)

        # Validation set: equal fair/unfair (if sweep mode)
        validation_clients = None
        if self.preferences.sweep and self.preferences.num_validation_nodes > 0:
            n_val = self.preferences.num_validation_nodes
            n_val_per_type = n_val // 2

            val_fair = remaining_fair[:n_val_per_type]
            val_unfair = remaining_unfair[:n_val_per_type]
            validation_clients = val_fair + val_unfair

            remaining_fair = remaining_fair[n_val_per_type:]
            remaining_unfair = remaining_unfair[n_val_per_type:]

        # Training set: all remaining clients (should be equal fair/unfair)
        train_clients = remaining_fair + remaining_unfair

        return train_clients, test_clients, validation_clients

    def _setup_cross_device(self) -> None:
        """
        Override cross-device setup to ensure equal fair/unfair distribution
        in train, validation, and test sets.
        """
        self.validation_clients_list = None

        # Sort clients for determinism
        self.clients_list = [
            str(client_id)
            for client_id in sorted([int(client_id) for client_id in self.clients_list])
        ]

        # Split with equal fair/unfair distribution
        train_clients, test_clients, validation_clients = self._split_clients_equal(
            len(self.clients_list)
        )

        self.test_clients_list = test_clients
        self.validation_clients_list = validation_clients
        self.training_clients_list = train_clients

        # Sample test clients per round (with equal fair/unfair per round)
        sampled_nodes_test = self.sample_clients_per_round(
            fraction=self.preferences.sampled_test_nodes_per_round,
            client_list=self.test_clients_list,
        )
        with open(f"{self.preferences.fed_dir}/test_nodes_per_round.pkl", "wb") as f:
            dill.dump(sampled_nodes_test, f)

        with open(f"{self.preferences.fed_dir}/test_nodes_list.pkl", "wb") as f:
            dill.dump(self.test_clients_list, f)

        log(INFO, f"Test clients list: {self.test_clients_list}")
        log(INFO, f"Sampled test clients per round: {sampled_nodes_test}")

        # Sample validation clients per round (if applicable)
        sampled_nodes_validation = None
        if self.validation_clients_list:
            sampled_nodes_validation = self.sample_clients_per_round(
                fraction=self.preferences.sampled_validation_nodes_per_round,
                client_list=self.validation_clients_list,
            )
            with open(
                f"{self.preferences.fed_dir}/validation_nodes_per_round.pkl",
                "wb",
            ) as f:
                dill.dump(sampled_nodes_validation, f)

            with open(
                f"{self.preferences.fed_dir}/validation_nodes_list.pkl", "wb"
            ) as f:
                dill.dump(self.validation_clients_list, f)

            log(INFO, f"Validation clients list: {self.validation_clients_list}")
            log(
                INFO,
                f"Sampled validation clients per round: {sampled_nodes_validation}",
            )

        # Sample training clients per round (with equal fair/unfair per round)
        sampled_nodes_train = self.sample_clients_per_round(
            fraction=self.preferences.sampled_training_nodes_per_round,
            client_list=self.training_clients_list,
        )
        log(INFO, f"Training clients list: {self.training_clients_list}")
        log(INFO, f"Sampled training clients per round: {sampled_nodes_train}")
        with open(f"{self.preferences.fed_dir}/train_nodes_per_round.pkl", "wb") as f:
            dill.dump(sampled_nodes_train, f)

        with open(f"{self.preferences.fed_dir}/train_nodes_list.pkl", "wb") as f:
            dill.dump(self.training_clients_list, f)

        log(INFO, f"Training clients: {len(self.training_clients_list)} total")
        log(
            INFO,
            f"Validation clients: {len(self.validation_clients_list) if self.validation_clients_list else 0} total",
        )
        log(INFO, f"Test clients: {len(self.test_clients_list)} total")

        self._save_counter_sampling(sampled_nodes_train)

        random.seed(self.preferences.seed)

    def sample_clients_per_round(
        self, fraction: float, client_list: list[str]
    ) -> dict[int, list[str]]:
        """
        Samples clients using EQUAL fair/unfair sampling per round.

        Ensures the same number of fair and unfair clients are selected each round.

        Args:
            fraction: Fraction of clients to sample each round.
            client_list: List of client CIDs (which are now partition IDs).

        Returns:
            dict[int, list[str]]: Mapping from round number to list of sampled client CIDs.

        """
        sampled_nodes = {}

        # Stratify clients by type
        fair_clients, unfair_clients = self._stratify_clients(client_list)

        log(
            INFO,
            f"Sampling from {len(fair_clients)} fair, {len(unfair_clients)} unfair clients",
        )

        total_pop = len(client_list)
        total_sample = int(fraction * total_pop)

        if total_pop == 0:
            return {}

        n_fair_pop = len(fair_clients)
        n_unfair_pop = len(unfair_clients)

        # EQUAL sampling: same number from each group
        # Use the smaller population to determine sample size per group
        min_pop = min(n_fair_pop, n_unfair_pop)

        if min_pop == 0:
            # Only one type of clients exists
            n_sample_per_type = total_sample // 2
            if n_fair_pop > 0:
                n_fair_sample = min(n_sample_per_type * 2, n_fair_pop)
                n_unfair_sample = 0
            else:
                n_fair_sample = 0
                n_unfair_sample = min(n_sample_per_type * 2, n_unfair_pop)
        else:
            # Equal sampling from both groups
            # Each group contributes half the total sample
            n_sample_per_type = total_sample // 2
            n_fair_sample = min(n_sample_per_type, n_fair_pop)
            n_unfair_sample = min(n_sample_per_type, n_unfair_pop)

        log(
            INFO,
            f"Sampling {n_fair_sample} fair and {n_unfair_sample} unfair per round",
        )

        for fl_round in range(self.preferences.num_rounds):
            round_samples = []

            # Sample Fair clients (rotating through the list)
            if n_fair_pop > 0 and n_fair_sample > 0:
                start = (fl_round * n_fair_sample) % n_fair_pop
                end = start + n_fair_sample
                if end <= n_fair_pop:
                    round_samples.extend(fair_clients[start:end])
                else:
                    # Wrap around
                    round_samples.extend(fair_clients[start:])
                    round_samples.extend(fair_clients[: end - n_fair_pop])

            # Sample Unfair clients (rotating through the list)
            if n_unfair_pop > 0 and n_unfair_sample > 0:
                start = (fl_round * n_unfair_sample) % n_unfair_pop
                end = start + n_unfair_sample
                if end <= n_unfair_pop:
                    round_samples.extend(unfair_clients[start:end])
                else:
                    round_samples.extend(unfair_clients[start:])
                    round_samples.extend(unfair_clients[: end - n_unfair_pop])

            sampled_nodes[fl_round] = round_samples

        return sampled_nodes
