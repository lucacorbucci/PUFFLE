import os
import random
import threading
import time

import dill
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from FlowerFLTemplate.Utils.preferences import Preferences


class SimpleClientManager(ClientManager):
    """Provides a pool of available clients."""

    def __init__(self, preferences: Preferences | None = None) -> None:
        """
        Initializes a SimpleClientManager instance for managing Flower clients.

        Sets up client dictionaries, threading condition, preferences, and lists for training/validation/test clients.

        Args:
            preferences (Preferences, optional): Configuration preferences for the FL setup. Defaults to None.

        Returns:
            None

        """
        self.clients: dict[str, ClientProxy] = {}
        self._cv = threading.Condition()
        self.preferences = preferences
        self.clients_list: list[str] = []
        self.num_round_train = 0
        self.num_round_validation = 0
        self.num_round_test = 0
        self.training_clients_list: list[str] = []
        self.validation_clients_list: list[str] = []
        self.test_clients_list: list[str] = []

    def __len__(self) -> int:
        """
        Returns the total number of registered clients.

        Args:
            None

        Returns:
            int: The number of currently available clients.

        """
        return len(self.clients)

    def num_available(self, phase: str) -> int:
        """
        Returns the number of available clients for a specific phase.

        Supports phases: "training", "validation", "test".

        Args:
            phase (str): The phase to check availability for ("training", "validation", or "test").

        Returns:
            int: The number of available clients for the specified phase.

        Raises:
            KeyError: If phase is invalid.

        """
        if phase == "training":
            return len(self.training_clients_list)
        if phase == "validation":
            return len(self.validation_clients_list)
        return len(self.test_clients_list)

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """
        Blocks until the specified number of clients are available or timeout is reached.

        Args:
            num_clients (int): The minimum number of clients to wait for.
            timeout (int, optional): Maximum wait time in seconds. Defaults to 86400 (24 hours).

        Returns:
            bool: True if the required number of clients became available within timeout, False otherwise.

        """
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.clients) >= num_clients, timeout=timeout
            )

    def pre_sample_clients(
        self, fraction: float, client_list: list[str]
    ) -> dict[int, list[str]]:
        """
        Pre-samples clients for each round without shuffling, for deterministic sampling.

        Args:
            fraction (float): Fraction of clients to sample per round.
            client_list (list[str]): List of client IDs to sample from.

        Returns:
            dict[int, list[str]]: Dictionary mapping round numbers to lists of sampled client IDs.

        """
        sampled_nodes = {}
        nodes_to_sample = int(fraction * len(client_list))
        for fl_round in range(self.preferences.num_rounds):
            if fraction == 1.0:
                start = 0
                end = len(client_list)
            else:
                # 9*15 % 150
                start = fl_round * nodes_to_sample % len(client_list)
                # (9*15 + 15) % 150
                end = (fl_round * nodes_to_sample + nodes_to_sample) % len(client_list)
            if end == 0:
                end = len(client_list)
            sampled_nodes[fl_round] = client_list[start:end]
        return sampled_nodes

    def sample_clients_per_round(
        self, fraction: float, client_list: list[str]
    ) -> dict[int, list[str]]:
        """
        Samples clients for each federated learning round based on the fraction.

        Handles wrapping around the client list for sampling. Prints sampling details.

        Args:
            fraction (float): Fraction of clients to sample per round.
            client_list (list[str]): List of client IDs to sample from.

        Returns:
            dict[int, list[str]]: Dictionary mapping round numbers to lists of sampled client IDs.

        """
        sampled_nodes = {}
        nodes_to_sample = int(fraction * len(client_list))
        for fl_round in range(self.preferences.num_rounds):
            # number of nodes we have to select in each round
            if fraction == 1.0:
                start = 0
                end = len(client_list)
            else:
                start = fl_round * nodes_to_sample % len(client_list)
                end = (fl_round * nodes_to_sample + nodes_to_sample) % len(client_list)
            if start < end:
                sampled_nodes[fl_round] = client_list[start:end]
            else:
                sampled_nodes[fl_round] = client_list[start:] + client_list[:end]

        return sampled_nodes

    def register(self, client: ClientProxy) -> bool:
        """
        Registers a Flower ClientProxy instance with the manager.

        Assigns a unique random CID if needed, handles cross-device/silo sampling,
        saves sampled nodes to pickle files, and notifies waiting threads.

        Args:
            client (ClientProxy): The client to register.

        Returns:
            bool: True if registration succeeded, False if already registered.

        Raises:
            IOError: If pickle file writing fails.

        """
        if client.cid in self.clients:
            return False

        self._assign_unique_cid(client)

        self.clients[client.cid] = client
        self.clients_list.append(client.cid)

        if self.preferences.num_clients == len(self.clients_list):
            self._setup_simulation_sets()

            with self._cv:
                self._cv.notify_all()

        return True

    def _assign_unique_cid(self, client: ClientProxy) -> None:
        new_random_cid = str(random.randint(0, self.preferences.num_clients))
        while new_random_cid in self.clients:
            new_random_cid = str(random.randint(0, self.preferences.num_clients))
        client.cid = new_random_cid

    def _setup_simulation_sets(self) -> None:
        if self.preferences.cross_device:
            self._setup_cross_device()
        else:
            self._setup_cross_silo()

    def _setup_cross_device(self) -> None:
        # In the cross device case, we want to sample the clients so that we have
        # a training set of clients and a test set of clients.
        # The test set of clients should always be the same both during the training and the testing.
        self.validation_clients_list = None
        self.clients_list = [
            str(client_id)
            for client_id in sorted([int(client_id) for client_id in self.clients_list])
        ]

        # sample the test clients from the self.clients_list
        self.test_clients_list = self.clients_list[: self.preferences.num_test_nodes]
        sampled_nodes_test = self.sample_clients_per_round(
            fraction=self.preferences.sampled_test_nodes_per_round,
            client_list=self.test_clients_list,
        )

        with open(f"{self.preferences.fed_dir}/test_nodes_per_round.pkl", "wb") as f:
            dill.dump(sampled_nodes_test, f)

        with open(f"{self.preferences.fed_dir}/test_nodes_list.pkl", "wb") as f:
            dill.dump(self.test_clients_list, f)

        remaining_nodes = self.clients_list[self.preferences.num_test_nodes :]

        random.seed(self.preferences.node_shuffle_seed)
        random.shuffle(remaining_nodes)

        # Now we check if we need to create the validation set
        if self.preferences.sweep and self.preferences.num_validation_nodes > 0:
            self.validation_clients_list = remaining_nodes[
                : self.preferences.num_validation_nodes
            ]
            remaining_nodes = remaining_nodes[self.preferences.num_validation_nodes :]
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

        self.training_clients_list = remaining_nodes

        sampled_nodes_train = self.sample_clients_per_round(
            fraction=self.preferences.sampled_training_nodes_per_round,
            client_list=self.training_clients_list,
        )
        with open(f"{self.preferences.fed_dir}/train_nodes_per_round.pkl", "wb") as f:
            dill.dump(sampled_nodes_train, f)

        with open(f"{self.preferences.fed_dir}/train_nodes_list.pkl", "wb") as f:
            dill.dump(self.training_clients_list, f)

        self._save_counter_sampling(sampled_nodes_train)

        random.seed(self.preferences.seed)

    def _setup_cross_silo(self) -> None:
        # In this case I'm in the cross-silo case
        # This means that each node has training, validation and test data
        # so each node could be used for training, validation and testing
        if self.preferences.sampled_validation_nodes_per_round:
            random.seed(self.preferences.node_shuffle_seed)
            random.shuffle(self.clients_list)
            sampled_nodes_validation = self.pre_sample_clients(
                fraction=self.preferences.sampled_validation_nodes_per_round,
                client_list=self.clients_list,
            )
            with open(
                f"{self.preferences.fed_dir}/validation_nodes_per_round.pkl",
                "wb",
            ) as f:
                dill.dump(sampled_nodes_validation, f)
            random.seed(self.preferences.seed)
        else:
            pass

        sampled_nodes_test = self.pre_sample_clients(
            fraction=self.preferences.sampled_test_nodes_per_round,
            client_list=self.clients_list,
        )

        with open(f"{self.preferences.fed_dir}/test_nodes_per_round.pkl", "wb") as f:
            dill.dump(sampled_nodes_test, f)

        sampled_nodes_train = self.pre_sample_clients(
            fraction=self.preferences.sampled_training_nodes_per_round,
            client_list=self.clients_list,
        )
        with open(f"{self.preferences.fed_dir}/train_nodes_per_round.pkl", "wb") as f:
            dill.dump(sampled_nodes_train, f)

        self._save_counter_sampling(sampled_nodes_train)

        self.test_clients_list = self.clients_list
        self.training_clients_list = self.clients_list
        self.validation_clients_list = self.clients_list

    def _save_counter_sampling(self, sampled_nodes_train: dict[int, list[str]]) -> None:
        counter_sampling: dict[str, int] = {}
        for sample_list in sampled_nodes_train.values():
            for node in sample_list:
                if node not in counter_sampling:
                    counter_sampling[str(node)] = 0
                counter_sampling[str(node)] += 1

        with open(f"{self.preferences.fed_dir}/counter_sampling.pkl", "wb") as f:
            dill.dump(counter_sampling, f)

    def unregister(self, client: ClientProxy) -> None:
        """
        Unregisters a Flower ClientProxy instance from the manager.

        Idempotent operation; notifies waiting threads if unregistered.

        Args:
            client (ClientProxy): The client to unregister.

        Returns:
            None

        """
        if client.cid in self.clients:
            del self.clients[client.cid]

            with self._cv:
                self._cv.notify_all()

    def all(self) -> dict[str, ClientProxy]:
        """
        Returns all registered clients.

        Args:
            None

        Returns:
            dict[str, ClientProxy]: Dictionary of all client IDs to ClientProxy instances.

        """
        return self.clients

    def sample(
        self,
        num_clients: int,
        phase: str,
        min_num_clients: int | None = None,
        criterion: Criterion | None = None,
    ) -> list[ClientProxy]:
        """
        Samples ClientProxy instances for a specific phase (training/validation/test).

        Waits for minimum clients, loads pre-sampled nodes from pickle files, and returns corresponding clients. Ignores num_clients and criterion parameters.

        Args:
            num_clients (int): Number of clients to sample (unused).
            phase (str): Phase for sampling ("training", "validation", or "test").
            min_num_clients (int | None, optional): Minimum clients to wait for. Defaults to num_clients.
            criterion (Criterion | None, optional): Sampling criterion (unused).

        Returns:
            list[ClientProxy]: List of sampled ClientProxy instances for the current round.

        Raises:
            IOError: If pickle file loading fails.

        """
        # Block until at least num_clients are connected.
        self.wait_for(num_clients)

        # wait until the file train_nodes_per_round.pkl is stored on disk

        def wait_file(file_path: str) -> None:
            """
            Waits until the specified file exists and is non-empty.

            Args:
                file_path (str): Path to the file to wait for.

            Returns:
                None

            """
            while True:
                if os.path.exists(file_path):
                    try:
                        if os.path.getsize(file_path) > 0:
                            break  # file exists and is non-empty
                    except OSError:
                        # file might be created but not fully ready yet
                        pass
                time.sleep(0.1)

        wait_file(f"{self.preferences.fed_dir}/train_nodes_per_round.pkl")
        wait_file(f"{self.preferences.fed_dir}/validation_nodes_per_round.pkl")
        wait_file(f"{self.preferences.fed_dir}/test_nodes_per_round.pkl")

        # Sample clients which meet the criterion

        if phase == "training":
            with open(
                f"{self.preferences.fed_dir}/train_nodes_per_round.pkl", "rb"
            ) as f:
                train_nodes = dill.load(f)

            sampled_clients = [self.clients[str(node)] for node in train_nodes[0]]
            self.num_round_train += 1

        elif phase == "validation":
            with open(
                f"{self.preferences.fed_dir}/validation_nodes_per_round.pkl", "rb"
            ) as f:
                validation_nodes = dill.load(f)

            sampled_clients = [
                self.clients[str(node)]
                for node in validation_nodes[self.num_round_validation]
            ]
            self.num_round_validation += 1
        else:
            with open(
                f"{self.preferences.fed_dir}/test_nodes_per_round.pkl", "rb"
            ) as f:
                test_nodes = dill.load(f)

            sampled_clients = [
                self.clients[str(node)] for node in test_nodes[self.num_round_test]
            ]
            self.num_round_test += 1

        return sampled_clients
