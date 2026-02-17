# ABOUTME: Integration test for MMD-Fair FL simulation.
# ABOUTME: Tests end-to-end Flower simulation with synthetic binary classification data.

import tempfile
from typing import Any, cast

import numpy as np
import torch
from datasets import Dataset
from FlowerFLTemplate.Models.utils import get_model
from FlowerFLTemplate.Utils.preferences import Preferences
from FlowerFLTemplate.Utils.utils import get_params
from flwr.common import ndarrays_to_parameters
from flwr_datasets.partitioner import IidPartitioner

from competitors.mmd_fair.simulation.client import MMDFairFlowerClient
from competitors.mmd_fair.simulation.strategy import MMDFairFedAvg


def create_synthetic_dataset(n_samples: int = 300, seed: int = 42) -> Dataset:
    """
    Create synthetic binary classification dataset with demographic groups.

    Similar to Fair-FL reference implementation.

    Args:
        n_samples: Number of samples to generate
        seed: Random seed

    Returns:
        HuggingFace Dataset with features, sensitive attribute, and target
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate features (5 dimensions)
    X = np.random.randn(n_samples, 5).astype(np.float32)

    # Generate binary sensitive attribute (balanced)
    Z = np.random.binomial(1, 0.5, n_samples).astype(np.int64)

    # Generate target with some correlation to features and demographic group
    # Y = sigmoid(X @ w + bias + demographic_bias * Z)
    w = np.random.randn(5).astype(np.float32)
    bias = 0.5
    demographic_bias = 0.3  # Slight bias based on demographic group

    logits = X @ w + bias + demographic_bias * Z
    probs = 1 / (1 + np.exp(-logits))
    Y = (probs > 0.5).astype(np.int64)

    # Create HuggingFace dataset
    data_dict = {
        "features": X.tolist(),
        "sensitive": Z.tolist(),
        "target": Y.tolist(),
    }

    return Dataset.from_dict(data_dict)


class TestMMDFairSimulation:
    """Integration tests for MMD-Fair FL simulation."""

    def test_client_initialization(self):
        """Test MMDFairFlowerClient initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preferences = Preferences(
                num_clients=3,
                num_rounds=2,
                cross_device=True,
                num_epochs=1,
                batch_size=32,
                lr=0.01,
                optimizer="sgd",
                momentum=0.9,
                weight_decay=1e-5,
                regularization_lambda=1.0,
                fed_dir=tmpdir,
                fl_setting="cross_device",
                dataset_name="synthetic",
                model="LinearClassificationNet",
                num_classes=1,
                in_channels=5,
            )

            # Create synthetic dataset
            dataset = create_synthetic_dataset(n_samples=100)
            partitioner = IidPartitioner(num_partitions=3)
            partitioner.dataset = dataset

            # Create data loader function
            def create_loader():
                from torch.utils.data import DataLoader, TensorDataset

                partition = partitioner.load_partition(0)
                df = cast(Any, partition.to_pandas())

                X = torch.tensor(np.array(df["features"].tolist()), dtype=torch.float32)
                Z = torch.tensor(df["sensitive"].values, dtype=torch.long)
                Y = torch.tensor(df["target"].values, dtype=torch.float32)

                dataset = TensorDataset(X, Z, Y)
                loader = DataLoader(dataset, batch_size=32, shuffle=True)
                return loader, loader

            # Create client
            client = MMDFairFlowerClient(
                partition_id=0,
                preferences=preferences,
                data_loader_fn=create_loader,
            )

            assert client.partition_id == 0
            assert client.preferences == preferences
            assert not client._initialized

    def test_strategy_initialization(self):
        """Test MMDFairFedAvg strategy initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            preferences = Preferences(
                num_clients=3,
                num_rounds=2,
                fed_dir=tmpdir,
                model="LinearClassificationNet",
                num_classes=1,
                in_channels=5,
            )

            # Create initial model
            model = get_model(
                model_name="LinearClassificationNet",
                num_classes=1,
                in_channels=5,
            )
            params = get_params(model)
            initial_params = ndarrays_to_parameters(params)

            # Create strategy
            strategy = MMDFairFedAvg(
                fraction_fit=1.0,
                fraction_evaluate=0.0,
                initial_parameters=initial_params,
                preferences=preferences,
                wandb_run=None,
                mu=1.0,
                ny=50,
                lambda_fairness=1.0,
            )

            assert strategy.mu == 1.0
            assert strategy.ny == 50
            assert strategy.lambda_fairness == 1.0
            assert strategy.Y_0 is None  # Not initialized until initialize_parameters
            assert strategy.Y_1 is None

    def test_prediction_tracker_lifecycle(self):
        """Test prediction tracker drop and update operations."""
        from competitors.mmd_fair.prediction_tracker import PredictionTracker

        tracker = PredictionTracker(demographic_group=0, capacity=100)

        # Initial state
        assert len(tracker) == 100
        assert tracker.get_predictions().shape == (100,)

        # Drop 50%
        tracker.drop(mu=0.5)
        assert len(tracker) < 100  # Should be around 50

        # Update with new predictions
        new_preds = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.4, 0.5])]
        tracker.update(new_preds)

        # Should have old (after drop) + new (5 elements)
        assert len(tracker) > 50

    def test_end_to_end_simulation_lightweight(self):
        """
        Lightweight end-to-end test of FL simulation.

        Tests that the simulation runs without errors for 2 rounds with 3 clients.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create synthetic dataset
            dataset = create_synthetic_dataset(n_samples=300, seed=42)
            partitioner = IidPartitioner(num_partitions=3)
            partitioner.dataset = dataset

            preferences = Preferences(
                num_clients=3,
                num_rounds=2,
                cross_device=True,
                num_epochs=1,
                sampled_training_nodes_per_round=1.0,
                sampled_validation_nodes_per_round=0.0,
                batch_size=32,
                lr=0.01,
                optimizer="sgd",
                momentum=0.9,
                weight_decay=1e-5,
                regularization_lambda=1.0,
                fed_dir=tmpdir,
                fl_setting="cross_device",
                dataset_name="synthetic",
                model="LinearClassificationNet",
                num_classes=1,
                in_channels=5,
            )

            # Add MMD-Fair specific attributes
            preferences.mu = 1.0  # type: ignore[attr-defined]
            preferences.ny = 50  # type: ignore[attr-defined]

            # Create initial model
            model = get_model(
                model_name="LinearClassificationNet",
                num_classes=1,
                in_channels=5,
            )
            initial_params_ndarrays = get_params(model)
            initial_params = ndarrays_to_parameters(initial_params_ndarrays)

            # Create strategy
            from competitors.mmd_fair.simulation.metrics import (
                aggregate_evaluate_metrics,
                aggregate_fit_metrics,
            )

            strategy = MMDFairFedAvg(
                fraction_fit=1.0,
                fraction_evaluate=0.0,
                initial_parameters=initial_params,
                preferences=preferences,
                wandb_run=None,
                mu=1.0,
                ny=50,
                lambda_fairness=1.0,
                fit_metrics_aggregation_fn=aggregate_fit_metrics,
                evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
            )

            # Initialize trackers
            from FlowerFLTemplate.ClientManager.client_manager import (
                SimpleClientManager,
            )

            client_manager = SimpleClientManager(preferences=preferences)
            strategy.initialize_parameters(client_manager)

            assert strategy.Y_0 is not None
            assert strategy.Y_1 is not None
            assert len(strategy.Y_0) == 50
            assert len(strategy.Y_1) == 50

            # Create clients
            clients = []
            for partition_id in range(3):

                def create_loader(pid=partition_id):
                    from torch.utils.data import DataLoader, Dataset

                    partition = partitioner.load_partition(pid)
                    df = cast(Any, partition.to_pandas())

                    X = torch.tensor(
                        np.array(df["features"].tolist()), dtype=torch.float32
                    )
                    Z = torch.tensor(df["sensitive"].values, dtype=torch.long)
                    Y = torch.tensor(df["target"].values, dtype=torch.float32)

                    # Create custom dataset matching DutchDataset format
                    class SyntheticDataset(Dataset):
                        def __init__(self, x, z, y):
                            self.x = x
                            self.z = z
                            self.y = y
                            self.indexes = list(range(len(x)))

                        def __len__(self):
                            return len(self.x)

                        def __getitem__(self, idx):
                            # Return 5-tuple: (x, z, y, index, idx)
                            return (
                                self.x[idx],
                                self.z[idx],
                                self.y[idx],
                                self.indexes[idx],
                                idx,
                            )

                    dataset = SyntheticDataset(X, Z, Y)
                    loader = DataLoader(dataset, batch_size=32, shuffle=True)
                    return loader, loader

                client = MMDFairFlowerClient(
                    partition_id=partition_id,
                    preferences=preferences,
                    data_loader_fn=create_loader,
                )
                clients.append(client)

            # Simulate 2 rounds
            for round_num in range(1, 3):
                # Each client trains
                fit_results = []
                for client in clients:
                    # Serialize Y_0/Y_1 for client
                    import io

                    Y_0_buffer = io.BytesIO()
                    Y_1_buffer = io.BytesIO()
                    torch.save(strategy.Y_0.get_predictions(), Y_0_buffer)
                    torch.save(strategy.Y_1.get_predictions(), Y_1_buffer)

                    config = {
                        "Y_0_bytes": Y_0_buffer.getvalue(),
                        "Y_1_bytes": Y_1_buffer.getvalue(),
                        "alpha_0": 1.0,
                        "alpha_1": 1.0,
                        "N": 300,
                    }

                    # Client fit
                    updated_params, num_examples, metrics = client.fit(
                        initial_params_ndarrays, config
                    )

                    # Check that predictions were sampled
                    assert "pred_0_bytes" in metrics
                    assert "pred_1_bytes" in metrics

                    # Mock FitRes
                    from flwr.common import FitRes

                    fit_res = FitRes(
                        status=None,  # type: ignore
                        parameters=ndarrays_to_parameters(updated_params),
                        num_examples=num_examples,
                        metrics=metrics,
                    )

                    # Mock ClientProxy
                    from unittest.mock import MagicMock

                    proxy = MagicMock()
                    proxy.cid = str(client.partition_id)

                    fit_results.append((proxy, fit_res))

                # Strategy aggregates
                aggregated_params, aggregated_metrics = strategy.aggregate_fit(
                    server_round=round_num,
                    results=fit_results,
                    failures=[],
                )

                # Check aggregation succeeded
                assert aggregated_params is not None

                # Check metrics aggregation
                # Note: Test uses mock fit results, so we manually check if metrics would be passed
                if "train_fairness" in metrics:
                    # In a real run, aggregate_fit_metrics would be called
                    # Here we just verify the client produced the metrics
                    assert "unfairness" in metrics
                    assert "fair_fl_p1" in metrics
                    assert "mmd_loss" in metrics

                # Check trackers were updated
                assert strategy.Y_0 is not None
                assert strategy.Y_1 is not None
                # Trackers should have predictions (may vary due to drop/update)
                assert len(strategy.Y_0) > 0
                assert len(strategy.Y_1) > 0

                # Verify Evaluate Metrics
                # Mock EvaluateRes
                from flwr.common import EvaluateRes

                val_metrics = {
                    "accuracy": 0.8,
                    "loss": 0.5,
                    "unfairness": 0.1,
                    "fair_fl_p1": 0.1,
                    "mmd_loss": 0.05,
                    "val_loss": 0.5,
                    "val_acc": 0.8,
                    "val_fairness": 0.1,
                    "val_fair_fl_p1": 0.1,
                    "val_mmd_loss": 0.05,
                }

                eval_res = EvaluateRes(
                    status=None,  # type: ignore
                    loss=0.5,
                    num_examples=100,
                    metrics=val_metrics,
                )

                proxy = MagicMock()
                proxy.cid = str(0)

                loss_aggregated, metrics_aggregated = strategy.aggregate_evaluate(
                    server_round=round_num, results=[(proxy, eval_res)], failures=[]
                )

                assert "val_fairness" in metrics_aggregated
                assert "val_fair_fl_p1" in metrics_aggregated
                assert "val_mmd_loss" in metrics_aggregated

                print(
                    f"Round {round_num}: Y_0={len(strategy.Y_0)}, Y_1={len(strategy.Y_1)}"
                )

            print("✓ End-to-end simulation completed successfully")
