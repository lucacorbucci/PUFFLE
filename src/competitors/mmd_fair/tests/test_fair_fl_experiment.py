# ABOUTME: Tests for Fair-FL experiment reproduction.
# ABOUTME: Verifies dataset loading, model architectures, and experiment runner.


import numpy as np
import pytest
import torch

from competitors.mmd_fair.simulation.fair_fl_datasets import CompasDataset
from competitors.mmd_fair.simulation.fair_fl_experiment import (
    P1,
    FairFLClient,
    FairFLServer,
    accuracy,
)
from competitors.mmd_fair.simulation.fair_fl_models import LogisticModel, TwoLayerNN


class TestFairFLModels:
    """Test Fair-FL model architectures."""

    def test_two_layer_nn_architecture(self):
        """Test TwoLayerNN matches Fair-FL architecture."""
        model = TwoLayerNN(input_size=8, hidden_size=16, output_size=1)

        # Check layers
        assert hasattr(model, "linear1")
        assert hasattr(model, "linear2")
        assert model.linear1.in_features == 8
        assert model.linear1.out_features == 16
        assert model.linear1.bias is not None  # Fair-FL uses bias=True
        assert model.linear2.in_features == 16
        assert model.linear2.out_features == 1
        assert model.linear2.bias is not None

    def test_two_layer_nn_forward(self):
        """Test TwoLayerNN forward pass."""
        model = TwoLayerNN(input_size=8)
        x = torch.randn(32, 8)
        output = model(x)
        assert output.shape == (32, 1)

    def test_logistic_model_architecture(self):
        """Test LogisticModel matches Fair-FL architecture."""
        model = LogisticModel(input_size=11, output_size=1)

        assert hasattr(model, "linear")
        assert model.linear.in_features == 11
        assert model.linear.out_features == 1
        assert model.linear.bias is not None  # Fair-FL uses bias=True

    def test_logistic_model_forward(self):
        """Test LogisticModel forward pass."""
        model = LogisticModel(input_size=11)
        x = torch.randn(32, 11)
        output = model(x)
        assert output.shape == (32, 1)


class TestFairFLDatasets:
    """Test Fair-FL dataset loaders."""

    def test_compas_dataset_loading(self):
        """Test COMPAS dataset loads and preprocesses correctly."""
        dataset = CompasDataset()
        datasets = dataset.load_data(homefolder="/home/lcorbucci/Fair-FL")

        # Should have ~3 clients (age categories)
        assert len(datasets) >= 2
        assert len(datasets) <= 4

        # Check each client's data
        for X, Y, A in datasets:
            # X should have 8 features (after one-hot encoding)
            assert X.shape[1] == 8

            # Y should be binary
            assert set(Y.unique()).issubset({0, 1})

            # A should be binary (African-American vs Caucasian)
            assert set(A.unique()).issubset({0.0, 1.0})

            # Should have samples
            assert len(X) > 0
            assert len(Y) == len(X)
            assert len(A) == len(X)


class TestFairFLMetrics:
    """Test Fair-FL metrics."""

    def test_accuracy_metric(self):
        """Test accuracy computation."""
        # Perfect predictions
        logits = torch.tensor([10.0, 10.0, -10.0, -10.0])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        acc = accuracy(logits, labels)
        assert acc == 1.0

        # 50% accuracy
        logits = torch.tensor([10.0, -10.0, 10.0, -10.0])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        acc = accuracy(logits, labels)
        assert acc == 0.5

    def test_p1_metric(self):
        """Test P1 fairness metric."""
        # Perfect fairness (equal positive rates)
        logits = torch.tensor([10.0, 10.0, 10.0, 10.0])
        sensitive = torch.tensor([0.0, 0.0, 1.0, 1.0])
        p1 = P1(logits, sensitive)
        assert p1 == 0.0

        # Maximum unfairness
        logits = torch.tensor([10.0, 10.0, -10.0, -10.0])
        sensitive = torch.tensor([0.0, 0.0, 1.0, 1.0])
        p1 = P1(logits, sensitive)
        assert p1 == 1.0


class TestFairFLExperiment:
    """Test Fair-FL experiment runner."""

    def test_client_initialization(self):
        """Test FairFLClient initialization."""
        import pandas as pd

        # Create synthetic data
        X = pd.DataFrame(np.random.randn(100, 8))
        Y = pd.Series(np.random.binomial(1, 0.5, 100))
        A = pd.Series(np.random.binomial(1, 0.5, 100).astype(float))

        model = TwoLayerNN(input_size=8)
        lossf = torch.nn.BCEWithLogitsLoss()

        client = FairFLClient(
            dataset=(X, Y, A),
            model=model,
            lossf=lossf,
            stepsize=0.01,
            batchsize=32,
            epochs=1,
            lambda_=1.0,
            device="cpu",
        )

        assert client.X.shape == (100, 8)
        assert client.Y.shape == (100,)
        assert client.A.shape == (100,)
        assert client.get_weight() == 100

    def test_client_train_test_split(self):
        """Test client train/test split."""
        import pandas as pd

        X = pd.DataFrame(np.random.randn(100, 8))
        Y = pd.Series(np.random.binomial(1, 0.5, 100))
        A = pd.Series(np.random.binomial(1, 0.5, 100).astype(float))

        model = TwoLayerNN(input_size=8)
        lossf = torch.nn.BCEWithLogitsLoss()

        client = FairFLClient(
            dataset=(X, Y, A),
            model=model,
            lossf=lossf,
            device="cpu",
        )

        client.split_train_test(test_size=0.25)

        # Check split sizes
        assert len(client.X) == 75
        assert client.X_test is not None
        assert len(client.X_test) == 25
        assert client.get_weight() == 75

    @pytest.mark.slow
    def test_server_one_round(self):
        """Test FairFLServer runs one training round."""
        import pandas as pd

        # Create synthetic datasets for 3 clients
        datasets = []
        for _ in range(3):
            X = pd.DataFrame(np.random.randn(100, 8))
            Y = pd.Series(np.random.binomial(1, 0.5, 100))
            A = pd.Series(np.random.binomial(1, 0.5, 100).astype(float))
            datasets.append((X, Y, A))

        # Create server
        server = FairFLServer(
            client_datasets=datasets,
            modelclass=lambda: TwoLayerNN(input_size=8),
            lossf=torch.nn.BCEWithLogitsLoss(),
            m=None,
            T=1,  # Just 1 round
            client_stepsize=0.01,
            client_batchsize=32,
            client_epochs=1,
            mu=1.0,
            NY=50,
            lambda_=1.0,
            datasetname="Test",
            runname="test",
            device="cpu",
        )

        server.train_test_split()
        server.sync_N()
        server.sync_Pa()

        # Run one round
        server.train()

        # Check that trackers were created
        assert server.Y_0 is not None
        assert server.Y_1 is not None

        # Check that we can test
        res, weights = server.test_current_model()
        assert len(res) == 3
        assert len(weights) == 3
