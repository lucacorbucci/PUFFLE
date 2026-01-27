import sys
from unittest.mock import MagicMock, patch

import pytest
from flwr.client import Client
from flwr.common import Context
from flwr.server import ServerAppComponents

# Adjust path to allow imports from FlowerFLTemplate src
sys.path.append("src")
# Need to inject preferences into main module namespace for client_fn/server_fn to work
# since they rely on global `preferences`
import FlowerFLTemplate.main as main_module
from FlowerFLTemplate.ClientManager.client_manager import (
    SimpleClientManager,
)

# Since we want to test main.py logic, we can import functions from main if possible
# or test components individually.
# Given complexity of main(), testing components (client_fn, server_fn) is better.
# We will test that client_fn and server_fn (imported from main) work as expected
# given initialized preferences.
from FlowerFLTemplate.main import (
    client_fn,
    prepare_data,
    server_fn,
)
from FlowerFLTemplate.Utils.preferences import Preferences


@pytest.fixture
def mock_preferences():
    return Preferences(
        num_clients=2,
        num_rounds=1,
        dataset_name="dutch",  # Use dutch to trigger specific logic
        fl_setting="cross_device",
        cross_device=True,  # Explicitly set according to fl_setting intention
        partitioner_type="iid",  # Set partitioner type
        batch_size=5,
        model="LinearClassificationNet",
        sampled_training_nodes_per_round=1.0,
        sampled_validation_nodes_per_round=0.0,
    )


@patch("FlowerFLTemplate.main.prepare_data_for_cross_device")
@patch("FlowerFLTemplate.main.partitioner")
def test_client_fn(mock_partitioner, mock_prepare, mock_preferences):
    """Test client_fn calls correct preparation function."""
    # Setup global preferences
    main_module.preferences = mock_preferences
    mock_partitioner.load_partition.return_value = MagicMock()

    # Mock context
    context = MagicMock(spec=Context)
    context.node_config = {"partition-id": "0"}

    # Mock return
    mock_prepare.return_value = MagicMock(spec=Client)

    # Run
    client = client_fn(context)

    # Assert
    mock_prepare.assert_called_once()
    assert client is not None


@patch("FlowerFLTemplate.main.get_model")
@patch("FlowerFLTemplate.main.get_params")
def test_server_fn(mock_get_params, mock_get_model, mock_preferences):
    """Test server_fn initializes components correctly."""
    # Setup global preferences and client_manager
    main_module.preferences = mock_preferences
    main_module.client_manager = MagicMock(spec=SimpleClientManager)
    main_module.wandb_run = None

    # Mock model and params
    mock_model = MagicMock()
    mock_get_model.return_value = mock_model
    mock_get_params.return_value = []  # lists of ndarrays

    # Mock context
    context = MagicMock(spec=Context)

    # Run
    components = server_fn(context)

    # Assert
    assert isinstance(components, ServerAppComponents)
    assert components.server is not None
    assert components.config is not None
    assert components.config.num_rounds == 1


import datasets


def test_prepare_data_dutch(mock_preferences):
    """Test prepare_data logic for dutch dataset."""
    # Ensure preferences is set in main_module
    main_module.preferences = mock_preferences

    with (
        patch("FlowerFLTemplate.main.get_data_info") as mock_get_info,
        patch("FlowerFLTemplate.main.load_dataset") as mock_load_dataset,
    ):
        # Mock get_data_info
        mock_get_info.return_value = {"scaler": MagicMock()}

        # Mock load_dataset
        mock_ds = MagicMock(spec=datasets.Dataset)
        mock_ds.__len__.return_value = 100  # Ensure bool(mock_ds) is True
        mock_load_dataset.return_value = {"train": mock_ds}

        partitioner = prepare_data(mock_preferences)

        mock_load_dataset.assert_called()
        assert partitioner is not None
