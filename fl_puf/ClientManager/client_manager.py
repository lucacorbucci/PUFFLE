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
"""Flower ClientManager."""

import os
import random
import threading
from logging import INFO
from typing import Dict, List, Optional

import dill
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion


class SimpleClientManager(ClientManager):
    """Provides a pool of available clients."""

    def __init__(
        self,
        num_clients,
        node_shuffle_seed: int,
        seed: int,
        sort_clients,
        num_training_nodes: int,
        num_validation_nodes: int,
        num_test_nodes: int,
        fed_dir: str,
    ) -> None:
        """Creates a SimpleClientManager.

        Parameters:
        -----------
        num_clients : int
            The pool size.
        sort_clients : bool
            If True, the clients will be sorted by their id.
        num_training_nodes : int
            The number of nodes to use for training.
        num_validation_nodes : int
            The number of nodes to use for validation.
        num_test_nodes : int
            The number of nodes to use for testing.
        node_shuffle_seed: int
            The seed to use for shuffling the nodes.
        seed: int
            The seed that we used to split the dataset
            and to assign nodes to the different sets.
        """
        self.seed = seed
        random.seed(self.seed)
        self.num_clients = num_clients
        self.clients: Dict[str, ClientProxy] = {}
        self.clients_list: List[str] = []
        self.training_clients_list: List[str] = []
        self.validation_clients_list: List[str] = []
        self.test_clients_list: List[str] = []
        self._cv = threading.Condition()
        self.current_index_training = 0
        self.current_index_validation = 0
        self.current_index_test = 0
        self.sort_clients = sort_clients
        self.num_training_nodes = num_training_nodes
        self.num_validation_nodes = num_validation_nodes
        self.num_test_nodes = num_test_nodes
        self.node_shuffle_seed = node_shuffle_seed
        self.fed_dir = fed_dir

    def __len__(self) -> int:
        return len(self.clients)

    def num_available(self, phase) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        if phase == "training":
            return len(self.training_clients_list)
        elif phase == "validation":
            return len(self.validation_clients_list)
        else:
            return len(self.test_clients_list)
        # return len(self)

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Wait until at least `num_clients` are available.

        Blocks until the requested number of clients is available or until a
        timeout is reached. Current timeout default: 1 day.

        Parameters
        ----------
        num_clients : int
            The number of clients to wait for.
        timeout : int
            The time in seconds to wait for, defaults to 86400 (24h).

        Returns
        -------
        success : bool
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.clients) >= num_clients, timeout=timeout
            )

    def register(self, client: ClientProxy) -> bool:
        """Register Flower ClientProxy instance.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if ClientProxy is
            already registered or can not be registered for any reason.
        """
        if client.cid in self.clients:
            return False

        self.clients[client.cid] = client
        # Add the client to the list of available clients and then
        # shuffle the list
        self.clients_list.append(client.cid)

        if self.num_clients == len(self.clients_list) and self.sort_clients:
            random.seed(self.seed)
            self.clients_list = sorted(self.clients_list)
            random.shuffle(self.clients_list)

            # We want to be sure that the nodes in the test set are always the same
            # so we shuffle the list of nodes and then we split the list into two parts
            # one part is the one of the nodes that will be used for training and
            # validation and the other one is the one of the nodes that
            #  will be used for testing.
            for node in self.clients_list[: self.num_test_nodes]:
                self.test_clients_list.append(node)
            self.remaining_nodes = self.clients_list[self.num_test_nodes :]
            print("Nodes in the test set: ", self.test_clients_list)

            # Since we want to split the clients into training, validation and test
            # set and we want to be sure that the nodes are selected the same
            # amount of times during the training, we split the list of nodes
            # into three parts. Then when we have to select the nodes, we make sure
            # that the nodes from the different sets are selected in a round robin
            # fashion. It is not the most elegant approach but given our requirements
            # it is probably the best one (and maybe the simplest and fastest way).

            # We shuffle the list of nodes and then we split the list into two parts
            random.seed(self.node_shuffle_seed)
            random.shuffle(self.remaining_nodes)

            for node in self.remaining_nodes[: self.num_training_nodes]:
                self.training_clients_list.append(node)
            print("Nodes in the training set: ", self.training_clients_list)
            for node in self.remaining_nodes[self.num_training_nodes :]:
                self.validation_clients_list.append(node)
            print("Nodes in the validation set: ", self.validation_clients_list)
            random.seed(self.seed)

        with self._cv:
            self._cv.notify_all()
        # import sys

        # sys.exit()
        return True

    def unregister(self, client: ClientProxy) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy
        """
        if client.cid in self.clients:
            del self.clients[client.cid]
            self.clients_list.remove(client.cid)

            with self._cv:
                self._cv.notify_all()

    def all(self) -> Dict[str, ClientProxy]:
        """Return all available clients."""
        return self.clients

    def sample(
        self,
        num_clients: int,
        phase: str,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion

        if phase == "training":
            sampled_clients = self.sample_clients(
                sampling_list=self.training_clients_list,
                current_index=self.current_index_training,
                num_clients=num_clients,
                criterion=criterion,
            )
            self.current_index_training += num_clients

            if os.path.exists(f"{self.fed_dir}/clients_last_round.pkl"):
                os.remove(f"{self.fed_dir}/clients_last_round.pkl")

            with open(f"{self.fed_dir}/clients_last_round.pkl", "wb") as f:
                dill.dump([client.cid for client in sampled_clients], f)

            print(
                "===>>>> Sampled for training: ",
                [client.cid for client in sampled_clients],
            )
        elif phase == "validation":
            sampled_clients = self.sample_clients(
                sampling_list=self.validation_clients_list,
                current_index=self.current_index_validation,
                num_clients=num_clients,
                criterion=criterion,
            )
            self.current_index_validation += num_clients
            print(
                "===>>>> Sampled for validation: ",
                [client.cid for client in sampled_clients],
            )
        else:
            sampled_clients = self.sample_clients(
                sampling_list=self.test_clients_list,
                current_index=self.current_index_test,
                num_clients=num_clients,
                criterion=criterion,
            )
            self.current_index_test += num_clients
            print(
                "===>>>> Sampled for test: ", [client.cid for client in sampled_clients]
            )
        return sampled_clients

    def sample_clients(
        self,
        sampling_list: List[str],
        current_index: int,
        num_clients: int,
        criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances considering the
        group of nodes to sample from.

        Parameters
        ----------
        sampling_list : list[str]
            The list of nodes to sample from.
        current_index : int
            The current index of the node to sample.
        num_clients : int
            The number of nodes to sample.
        criterion : Criterion
            The criterion to use for sampling.

        Returns
        -------
        sampled_clients : list[ClientProxy]
            The list of sampled nodes.
        """
        available_cids = sampling_list
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        # If we are training, we want to sample the nodes
        # from the training set of nodes.
        # We also want to be sure that we sample the clients
        # the same amount of times during the training.
        start = current_index % len(available_cids)
        end = (current_index + num_clients) % len(available_cids)

        if end > start:
            sampled_cids = available_cids[start:end]
        else:
            sampled_cids = available_cids[start:] + available_cids[:end]

        return [self.clients[cid] for cid in sampled_cids]
