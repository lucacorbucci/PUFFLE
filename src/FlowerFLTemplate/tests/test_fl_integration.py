import sys
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# Adjust path to allow imports from FlowerFLTemplate
# This is needed because the template is not a proper package installed in site-packages
sys.path.append("src/FlowerFLTemplate")

from FlowerFLTemplate.main import main as fl_main


@pytest.fixture
def mock_dataset():
    """Create dummy dataset loaders."""
    x = torch.randn(10, 10)
    z = torch.randint(0, 2, (10,))
    y = torch.randint(0, 2, (10,))
    dataset = TensorDataset(x, z, y)
    loader = DataLoader(dataset, batch_size=5)

    # Mock partitions structure: {0: {'train': loader, 'validation': loader}, ...}
    partitions = {
        0: {"train": loader, "validation": loader},
        1: {"train": loader, "validation": loader},
    }
    return partitions


@patch("main.load_partitioned_dataset")
@patch("main.fl.simulation.start_simulation")
@patch("main.wandb")
@patch("Client.client.dill.load")  # Mock pickle loading in client
@patch("builtins.open")  # Mock open for pickle
def test_fl_simulation_setup(
    mock_open, mock_dill, mock_wandb, mock_start_simulation, mock_load, mock_dataset
):
    """Test that main.py sets up and calls start_simulation correctly."""
    # Mock logic
    mock_load.return_value = mock_dataset
    mock_dill.return_value = {}  # Empty counter_sampling

    # Mock generic open to avoid file not found for counter_sampling.pkl
    mock_open.return_value.__enter__.return_value = MagicMock()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock command line arguments
        test_args = [
            "main.py",
            "--dataset_name",
            "dummy",
            "--num_clients",
            "2",
            "--num_rounds",
            "1",
            "--batch_size",
            "5",
            "--project_name",
            "TestFL",  # Use wandb=False but just in case
            "--sampled_training_nodes_per_round",
            "1.0",
            "--sampled_validation_nodes_per_round",
            "0.0",
            "--fed_dir",
            temp_dir,
            "--fairness_metric",
            "disparity",
            "--unfairness_reduction",
            "True",
            "--model",
            "LinearClassificationNet",
            "--in_channels",
            "10",
            "--num_classes",
            "2",
        ]

        with patch.object(sys, "argv", test_args):
            # Run main
            fl_main()

    # Verify dataset was loaded
    mock_load.assert_called_once()

    # Verify simulation started
    mock_start_simulation.assert_called_once()

    # Verify args passed to simulation
    call_kwargs = mock_start_simulation.call_args[1]
    assert call_kwargs["num_clients"] == 2  # noqa: S101
    assert call_kwargs["config"].num_rounds == 1  # noqa: S101
    assert call_kwargs["client_fn"] is not None  # noqa: S101

    # Test client_fn (simulate one client creation)
    client_fn = call_kwargs["client_fn"]
    client = client_fn("0")

    assert client is not None  # noqa: S101
    assert client.partition_id == 0  # noqa: S101
    # Verify initialization of PUFFLEModel wrapper
    assert client.model is not None  # noqa: S101
    assert client.model.config.lambda_regularization == 0.0  # noqa: S101 Default unless set
