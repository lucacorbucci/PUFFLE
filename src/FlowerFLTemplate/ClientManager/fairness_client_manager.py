import random
import threading
import time

from flwr.server.client_proxy import ClientProxy

from FlowerFLTemplate.ClientManager.client_manager import SimpleClientManager
from FlowerFLTemplate.Utils.preferences import Preferences


class FairnessClientManager(SimpleClientManager):
    """
    ClientManager that preserves client CIDs (for simulation) and samples clients
    based on their fairness type (fair/unfair) to ensure proportional representation.
    """

    def __init__(self, preferences: Preferences, client_types: dict[int, str]) -> None:
        """
        Args:
            preferences: Configuration preferences.
            client_types: Dictionary mapping partition ID (int) to type ("fair" or "unfair").

        """
        super().__init__(preferences)
        self.client_types = client_types

    # def _assign_unique_cid(self, client: ClientProxy) -> None:
    #     """
    #     Override to preserve CID.
    #     In simulation, CIDs are '0', '1', etc. which map to partition IDs.
    #     We do NOT want to randomize them.
    #     """
    #     # If CID is already in use, we might have a collision logic,
    #     # but for this purpose we assume unique CIDs are provided by simulation.
    #     # We perform NO operation here to keep client.cid as is.

    def sample_clients_per_round(
        self, fraction: float, client_list: list[str]
    ) -> dict[int, list[str]]:
        """
        Samples clients effectively using stratified sampling per round.
        Ensures fair and unfair clients are selected proportionally to their population size.
        """
        print("client_list: ", client_list)
        sampled_nodes = {}

        # Stratify
        fair_clients = []
        unfair_clients = []

        for cid in client_list:
            # Assume CID can be cast to int to lookup type
            try:
                cid_int = int(cid)
                c_type = self.client_types.get(cid_int, "unknown")
                if c_type == "fair":
                    fair_clients.append(cid)
                elif c_type == "unfair":
                    unfair_clients.append(cid)
                else:
                    # Default to fair or separate? Let's treat unknown as fair for now or log warning
                    fair_clients.append(cid)
            except ValueError:
                # String CID not int?
                fair_clients.append(cid)

        # Sort for determinism
        fair_clients.sort(key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
        unfair_clients.sort(key=lambda x: (0, int(x)) if x.isdigit() else (1, x))

        print(f"Fair clients: {fair_clients}")
        print(f"Unfair clients: {unfair_clients}")

        total_pop = len(client_list)
        total_sample = int(fraction * total_pop)

        if total_pop == 0:
            return {}

        n_fair_pop = len(fair_clients)
        n_unfair_pop = len(unfair_clients)

        # Calculate allocations (proportional)
        # We use round() or int() depending on preference. int() floors.
        if n_fair_pop + n_unfair_pop > 0:
            n_fair_sample = int(total_sample * (n_fair_pop / total_pop))
            n_unfair_sample = total_sample - n_fair_sample
        else:
            n_fair_sample = 0
            n_unfair_sample = 0

        for fl_round in range(self.preferences.num_rounds):
            round_samples = []

            # Sample Fair
            if n_fair_pop > 0 and n_fair_sample > 0:
                start = (fl_round * n_fair_sample) % n_fair_pop
                end = start + n_fair_sample
                if end <= n_fair_pop:
                    round_samples.extend(fair_clients[start:end])
                else:
                    # Wrap around
                    round_samples.extend(fair_clients[start:])
                    round_samples.extend(fair_clients[: end - n_fair_pop])

            # Sample Unfair
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
