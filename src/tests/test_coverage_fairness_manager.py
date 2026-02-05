# ruff: noqa: S101, PLR2004, PT018, D102, ARG002
from unittest.mock import MagicMock

import pytest

from FlowerFLTemplate.ClientManager.fairness_client_manager import FairnessClientManager
from FlowerFLTemplate.Utils.preferences import Preferences


class TestFairnessManagerCoverage:
    @pytest.fixture
    def preferences(self, tmp_path):
        pref = MagicMock(spec=Preferences)
        pref.num_rounds = 2
        return pref

    def test_non_int_cid(self, preferences):
        """Test handling of non-integer CIDs."""
        client_types = {0: "fair"}
        manager = FairnessClientManager(preferences, client_types)

        # 'a' causes ValueError in int conversion
        client_list = ["0", "a"]

        # fraction 1.0 -> sample all
        samples = manager.sample_clients_per_round(1.0, client_list)

        # 'a' should be treated as fair (fallback)
        # Total 2 clients, if 'a' is fair, both are fair.
        assert len(samples[0]) == 2
        assert "a" in samples[0]

    def test_unknown_client_type(self, preferences):
        """Test handling of client with unknown type."""
        client_types = {0: "fair"}
        manager = FairnessClientManager(preferences, client_types)

        # '1' is not in client_types
        client_list = ["0", "1"]

        samples = manager.sample_clients_per_round(1.0, client_list)

        # '1' should be treated as fallback (fair)
        assert len(samples[0]) == 2
        assert "1" in samples[0]

    def test_empty_population(self, preferences):
        """Test sampling with empty client list."""
        manager = FairnessClientManager(preferences, {})
        samples = manager.sample_clients_per_round(1.0, [])
        assert samples == {}

    def test_wrap_around_logic(self, preferences):
        """Test wrap-around sampling when sample size causes index overflow."""
        # 3 fair clients: 0, 1, 2
        client_types = {0: "fair", 1: "fair", 2: "fair"}
        manager = FairnessClientManager(preferences, client_types)

        client_list = ["0", "1", "2"]

        # Sample 2 per round.
        # Round 0: 0, 1
        # Round 1: 2, 0 (Wrap around)

        samples = manager.sample_clients_per_round(0.7, client_list)

        assert len(samples[0]) == 2
        assert "0" in samples[0] and "1" in samples[0]

        assert len(samples[1]) == 2
        # Round 1 should be 2, 0
        assert "2" in samples[1] and "0" in samples[1]

    def test_unfair_wrap_around(self, preferences):
        # 3 unfair clients: 0, 1, 2
        client_types = {0: "unfair", 1: "unfair", 2: "unfair"}
        manager = FairnessClientManager(preferences, client_types)

        client_list = ["0", "1", "2"]

        # Sample 2 per round.
        samples = manager.sample_clients_per_round(0.7, client_list)

        assert len(samples[1]) == 2
        # Round 1 should be 2, 0
        assert "2" in samples[1] and "0" in samples[1]
