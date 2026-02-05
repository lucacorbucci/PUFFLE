from unittest.mock import MagicMock

import pytest
from flwr.server.client_proxy import ClientProxy

from FlowerFLTemplate.ClientManager.fairness_client_manager import FairnessClientManager
from FlowerFLTemplate.Utils.preferences import Preferences

# ruff: noqa: S101, PLR2004


class TestFairnessClientManager:
    @pytest.fixture
    def preferences(self, tmp_path):
        """Create a mock Preferences object."""
        pref = MagicMock(spec=Preferences)
        pref.num_clients = 10
        pref.num_rounds = 5
        pref.fed_dir = str(tmp_path)
        pref.cross_device = False  # Cross-silo for simplicity in test

        # Sampling fractions
        pref.sampled_training_nodes_per_round = 0.4  # 4 clients per round
        pref.sampled_validation_nodes_per_round = None
        pref.sampled_test_nodes_per_round = 1.0

        pref.node_shuffle_seed = 42
        pref.seed = 42
        return pref

    def test_cid_preservation(self, preferences):
        """Test that CIDs are preserved and not randomized."""
        client_types = {i: "fair" for i in range(10)}  # noqa: C420
        manager = FairnessClientManager(preferences, client_types)

        # Register client "0"
        c0 = MagicMock(spec=ClientProxy)
        c0.cid = "0"
        # Mock get_properties to return partition_id
        mock_props = {"partition_id": 0}
        c0.get_properties.return_value.properties = mock_props
        manager.register(c0)

        # The CID should be set to str(partition_id) = "0"
        assert c0.cid == "0"
        assert "0" in manager.clients

    def test_stratified_sampling(self, preferences):
        """Test that sampling includes fair and unfair clients proportionally."""
        # 10 clients: 0-4 Fair (5), 5-9 Unfair (5) -> 50/50
        client_types = {}
        for i in range(5):
            client_types[i] = "fair"
        for i in range(5, 10):
            client_types[i] = "unfair"

        manager = FairnessClientManager(preferences, client_types)

        # Register all
        for i in range(10):
            c = MagicMock(spec=ClientProxy)
            c.cid = str(i)
            manager.register(c)

        client_list = [str(i) for i in range(10)]
        sampled_schedule = manager.sample_clients_per_round(0.4, client_list)

        # Total sample size expected = 10 * 0.4 = 4.
        # Fair pop = 5 (50%), Unfair pop = 5 (50%).
        # Expected Fair sample = 4 * 0.5 = 2.
        # Expected Unfair sample = 2.

        for samples in sampled_schedule.values():
            assert len(samples) == 4
            fair_count = sum(1 for cid in samples if client_types[int(cid)] == "fair")
            unfair_count = sum(
                1 for cid in samples if client_types[int(cid)] == "unfair"
            )

            assert fair_count == 2
            assert unfair_count == 2

    def test_stratified_sampling_uneven(self, preferences):
        """Test stratified sampling with uneven population."""
        # 10 clients: 0-7 Fair (80%), 8-9 Unfair (20%)
        client_types = {}
        for i in range(8):
            client_types[i] = "fair"
        for i in range(8, 10):
            client_types[i] = "unfair"

        manager = FairnessClientManager(preferences, client_types)

        # Test direct sampling function
        client_list = [str(i) for i in range(10)]

        # Fraction 0.5 -> 5 clients total.
        # FairnessClientManager samples EQUAL numbers of fair/unfair
        # So with floor(5/2) = 2 each = 4 total (not 5)
        sampled_schedule = manager.sample_clients_per_round(0.5, client_list)

        for samples in sampled_schedule.values():
            assert len(samples) == 4  # 2 fair + 2 unfair
            fair_count = sum(1 for cid in samples if client_types[int(cid)] == "fair")
            unfair_count = sum(
                1 for cid in samples if client_types[int(cid)] == "unfair"
            )

            assert fair_count == 2
            assert unfair_count == 2
