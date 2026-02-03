import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset
from flwr.server.client_proxy import ClientProxy

# Ensure src is in path for imports if running from root
sys.path.append(str(Path("src").resolve()))

from FlowerFLTemplate.ClientManager.fairness_client_manager import FairnessClientManager
from FlowerFLTemplate.Datasets.Partitioner.fairness_partitioner import (
    FairnessPartitioner,
)
from FlowerFLTemplate.Utils.preferences import Preferences


class TestFairnessIntegration:
    @pytest.fixture
    def dataset(self):
        """Creates a dummy dataset for testing."""
        num_samples = 1000
        rng = np.random.default_rng(42)
        target = rng.integers(0, 2, num_samples)
        sensitive = rng.integers(0, 2, num_samples)
        df = pd.DataFrame(
            {"target": target, "sensitive": sensitive, "data": range(num_samples)}
        )
        return Dataset.from_pandas(df)

    @pytest.fixture
    def preferences(self, tmp_path):
        """Creates a real Preferences object."""
        return Preferences(
            num_clients=10,
            num_rounds=5,
            fed_dir=str(tmp_path / "fed_dir"),
            cross_device=False,
            sampled_training_nodes_per_round=0.4,
            sampled_validation_nodes_per_round=None,
            sampled_test_nodes_per_round=1.0,
            node_shuffle_seed=42,
            seed=42,
        )

    def test_partitioner_client_manager_integration(self, dataset, preferences):
        """Verifies the end-to-end flow from Partitioner to ClientManager sampling."""
        # 1. Partition Data
        num_partitions = 10
        expected_fair = 5
        expected_unfair = 5

        partitioner = FairnessPartitioner(
            num_partitions=num_partitions,
            sensitive_attribute="sensitive",
            target_attribute="target",
            ratio_unfair_clients=0.5,  # 5 Fair, 5 Unfair
            group_to_reduce=(1, 0),
            ratio_unfairness=(0.8, 0.9),
            seed=42,
            dataset=dataset,
        )

        # Validation of partitioner output
        fair_nodes = [k for k, v in partitioner.client_types.items() if v == "fair"]
        unfair_nodes = [k for k, v in partitioner.client_types.items() if v == "unfair"]

        assert len(fair_nodes) == expected_fair, (
            f"Expected {expected_fair} fair nodes, got {len(fair_nodes)}"
        )  # noqa: S101
        assert len(unfair_nodes) == expected_unfair, (  # noqa: S101
            f"Expected {expected_unfair} unfair nodes, got {len(unfair_nodes)}"
        )

        # 2. Setup Client Manager
        # Ensure fed_dir exists
        fed_dir_path = Path(preferences.fed_dir)
        if not fed_dir_path.exists():
            fed_dir_path.mkdir(parents=True)

        manager = FairnessClientManager(preferences, partitioner.client_types)

        # Register clients
        for i in range(num_partitions):
            c = MagicMock(spec=ClientProxy)
            c.cid = str(i)
            manager.register(c)

        # 3. Run Sampling and Verify Stratification
        fraction = 0.4
        client_list = [str(i) for i in range(num_partitions)]
        schedule = manager.sample_clients_per_round(fraction, client_list)

        expected_sample_count = 2

        for fl_round, cids in schedule.items():
            types = [partitioner.client_types[int(cid)] for cid in cids]
            n_fair = types.count("fair")
            n_unfair = types.count("unfair")

            # With 50/50 population and 4 samples, we expect 2 Fair and 2 Unfair
            assert n_fair == expected_sample_count, (
                f"Round {fl_round}: Expected {expected_sample_count} Fair, got {n_fair}"
            )  # noqa: S101
            assert n_unfair == expected_sample_count, (
                f"Round {fl_round}: Expected {expected_sample_count} Unfair, got {n_unfair}"
            )  # noqa: S101
