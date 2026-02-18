# ABOUTME: Standalone experiment runner reproducing Fair-FL paper experiments.
# ABOUTME: Uses Fair-FL's data loading with PUFFLE's MMDFairModel for exact comparison.

import argparse
import os
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import wandb
from competitors.mmd_fair.model import MMDFairModel
from competitors.mmd_fair.prediction_tracker import PredictionTracker
from competitors.mmd_fair.simulation.fair_fl_datasets import CompasDataset
from competitors.mmd_fair.simulation.fair_fl_models import TwoLayerNN


def accuracy(p: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute accuracy from logits and labels.

    Args:
        p: Predicted logits
        y: True labels

    Returns:
        Accuracy as float
    """
    return (torch.sigmoid(p).round().flatten() == y).float().mean().item()


def P1(p: torch.Tensor, a: torch.Tensor) -> float:
    """
    Compute P1 fairness metric from Fair-FL.

    P1 = |P(Y_hat=1|A=0) - P(Y_hat=1|A=1)|

    Args:
        p: Predicted logits
        a: Sensitive attribute

    Returns:
        P1 metric as float
    """
    pred = torch.sigmoid(p).round().flatten()
    p_y1_a0 = (
        (pred[a == 0] == 1).float().mean() if (a == 0).any() else torch.tensor(0.0)
    )
    p_y1_a1 = (
        (pred[a == 1] == 1).float().mean() if (a == 1).any() else torch.tensor(0.0)
    )
    return abs(p_y1_a0 - p_y1_a1).item()


class FairFLClient:
    """
    Client abstraction matching Fair-FL's client.py but using PUFFLE's MMDFairModel.
    """

    def __init__(
        self,
        dataset: tuple[pd.DataFrame, pd.Series, pd.Series],
        model: torch.nn.Module,
        lossf: torch.nn.Module,
        stepsize: float = 0.1,
        batchsize: int = 100,
        epochs: int = 10,
        lambda_: float = 1.0,
        device: str = "cpu",
    ):
        """
        Initialize Fair-FL client.

        Args:
            dataset: (X, Y, A) tuple
            model: Neural network model
            lossf: Loss function
            stepsize: Learning rate
            batchsize: Batch size
            epochs: Number of local epochs
            lambda_: Fairness regularization weight
            device: Device for computation
        """
        X, Y, A = dataset
        self.X = torch.tensor(X.to_numpy(), device=device).float()
        self.Y = torch.tensor(Y.to_numpy(), device=device).float()
        self.A = torch.tensor(A.to_numpy(), device=device).float()

        self.stepsize = stepsize
        self.batchsize = batchsize
        self.epochs = epochs
        self.device = device

        # Create optimizer
        self.optimizer = torch.optim.SGD(model.parameters(), lr=stepsize)

        # Create MMDFairModel with default config, then override lambda
        # We can't use PUFFLEConfig here because it has le=1.0 validation,
        # but Fair-FL uses lambda values up to 100
        from puffle.Utils.config import PUFFLEConfig

        config = PUFFLEConfig()  # Use defaults
        self.model = MMDFairModel(
            model=model,
            optimizer=self.optimizer,
            criterion=lossf,
            device=device,
            config=config,
        )
        # Override lambda directly (bypassing validation)
        self.model.lambda_regularization = lambda_

        # Test split
        self.X_test: torch.Tensor | None = None
        self.Y_test: torch.Tensor | None = None
        self.A_test: torch.Tensor | None = None

        # Alpha weights (set by server)
        self.alphak0 = 1.0
        self.alphak1 = 1.0

        # Total samples N (set by server)
        self.N = 1

    def split_train_test(self, test_size: float = 0.25) -> None:
        """Split data into train/test sets."""
        (
            self.X,
            self.X_test,
            self.Y,
            self.Y_test,
            self.A,
            self.A_test,
        ) = train_test_split(
            self.X, self.Y, self.A, test_size=test_size, random_state=42
        )

    def get_weight(self) -> int:
        """Get client weight (number of training samples)."""
        return len(self.Y)

    def get_Pka(self, a: int = 0) -> float:
        """Get proportion of samples with A=a."""
        return (self.A == a).to(float).mean().cpu().item()

    def set_alphaka(self, Pa0: float) -> None:
        """Set alpha weights based on global P(A=0)."""
        self.alphak0 = self.get_Pka(a=0) / Pa0
        self.alphak1 = self.get_Pka(a=1) / (1 - Pa0)

    def set_N(self, N: int) -> None:
        """Set total sample count."""
        self.N = N
        self.model.set_total_samples(N)

    def set_C(self, Y_0: PredictionTracker, Y_1: PredictionTracker) -> None:
        """Set tracking function from server's Y_0/Y_1."""
        y0_tensor = Y_0.get_predictions()
        y1_tensor = Y_1.get_predictions()
        self.model.set_server_predictions(y0_tensor, y1_tensor)
        self.model.set_client_weights(self.alphak0, self.alphak1)
        self.model.set_tracking_function(y0_tensor, y1_tensor)

    def client_step(
        self, current_theta: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        Perform local training.

        Args:
            current_theta: Global model parameters

        Returns:
            Updated model parameters
        """
        # Load global parameters
        self.model.model.load_state_dict(current_theta)

        # Create dataloader
        if self.batchsize is not None:
            dataset = TensorDataset(self.X, self.A, self.Y)
            dataloader = DataLoader(dataset, batch_size=self.batchsize, shuffle=True)
        else:
            # Full batch
            dataset = TensorDataset(self.X, self.A, self.Y)
            dataloader = DataLoader(dataset, batch_size=len(self.X), shuffle=False)

        # Train for local epochs
        self.model.model.train()
        for _ in range(self.epochs):
            for batch in dataloader:
                # Batch is (x, a, y) but _train_batch expects (x, z, y)
                # where z is the sensitive attribute
                x_batch, a_batch, y_batch = batch
                batch_reordered = (x_batch, a_batch, y_batch)
                self.model._train_batch(
                    batch_reordered,
                    self.model.model,
                    self.optimizer,
                    self.model.criterion,
                )

        # Learning rate decay (0.99× per round, matching Fair-FL)
        self.stepsize *= 0.99
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.stepsize

        return self.model.model.state_dict()

    def sample_C_update(
        self, num_points: list[int], current_theta: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample predictions for server's update_C.

        Args:
            num_points: [num_samples_A0, num_samples_A1]
            current_theta: Current global model parameters

        Returns:
            (predictions_A0, predictions_A1)
        """
        with torch.no_grad():
            self.model.model.load_state_dict(current_theta)
            self.model.model.eval()

            # Sample indices
            n_a1 = int(self.A.sum())
            n_a0 = int((1 - self.A).sum())

            n_sample_a1 = int(self.alphak1 * num_points[1])
            n_sample_a0 = int(self.alphak0 * num_points[0])

            if n_a1 > 0 and n_sample_a1 > 0:
                point_idxs_a1 = np.random.choice(
                    n_a1, size=min(n_sample_a1, n_a1), replace=False
                )
                p1 = self.model.model(self.X[self.A == 1][point_idxs_a1])
                p1 = torch.sigmoid(p1).squeeze()
            else:
                p1 = torch.tensor([], device=self.device)

            if n_a0 > 0 and n_sample_a0 > 0:
                point_idxs_a0 = np.random.choice(
                    n_a0, size=min(n_sample_a0, n_a0), replace=False
                )
                p0 = self.model.model(self.X[self.A == 0][point_idxs_a0])
                p0 = torch.sigmoid(p0).squeeze()
            else:
                p0 = torch.tensor([], device=self.device)

            return p0, p1

    def test_client(
        self, theta: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Test client on test set.

        Args:
            theta: Model parameters

        Returns:
            (predictions, labels, sensitive_attributes)
        """
        with torch.no_grad():
            self.model.model.load_state_dict(theta)
            self.model.model.eval()
            preds = self.model.model(self.X_test)
            return preds, self.Y_test, self.A_test


class FairFLServer:
    """
    Server abstraction matching Fair-FL's server.py but using PUFFLE's components.
    """

    def __init__(
        self,
        client_datasets: list[tuple],
        modelclass: Callable[[], torch.nn.Module],
        lossf: torch.nn.Module,
        m: int | None = None,
        T: int = 50,
        client_stepsize: float = 5e-2,
        client_batchsize: int = 100,
        client_epochs: int = 10,
        mu: float = 1.0,
        NY: int = 100,
        lambda_: float = 1.0,
        datasetname: str = "None",
        runname: str = "",
        device: str = "cpu",
        convergence: bool = False,
        additional_config: dict[str, Any] | None = None,
    ):
        """
        Initialize Fair-FL server.

        Args:
            client_datasets: List of (X, Y, A) tuples
            modelclass: Model class to instantiate
            lossf: Loss function
            m: Number of clients per round (None = all)
            T: Number of communication rounds
            client_stepsize: Client learning rate
            client_batchsize: Client batch size
            client_epochs: Client local epochs
            mu: Drop rate for prediction tracking
            NY: Capacity of prediction trackers
            lambda_: Fairness regularization weight
            datasetname: Dataset name for logging
            runname: Run name for wandb
            device: Device for computation
            convergence: Whether to log convergence metrics
            additional_config: Additional wandb config
        """
        self.m = len(client_datasets) if m is None else m
        self.T = T
        self.mu = mu
        self.NY = NY
        self.device = device
        self.convergence = convergence

        # Create clients
        self.clients = [
            FairFLClient(
                dataset,
                modelclass(),
                lossf,
                stepsize=client_stepsize,
                batchsize=client_batchsize,
                epochs=client_epochs,
                lambda_=lambda_,
                device=device,
            )
            for dataset in client_datasets
        ]

        # Client weights
        self.client_weights = np.array([client.get_weight() for client in self.clients])
        self.client_weights = self.client_weights / self.client_weights.sum()

        # Global model
        self.model = modelclass().to(device)

        # Prediction trackers
        self.Y_0: PredictionTracker | None = None
        self.Y_1: PredictionTracker | None = None

        # Wandb config
        config = {
            "m": m,
            "T": T,
            "client_epochs": client_epochs,
            "client_stepsize": client_stepsize,
            "client_batchsize": client_batchsize,
            "mu": mu,
            "NY": NY,
            "lambda_": lambda_,
            "dataset": datasetname,
            "runname": runname,
        }
        if additional_config:
            config.update(additional_config)

        # Initialize wandb
        project = "fairFLConvergence" if convergence else "fairFL"
        wandb.init(project=project, config=config)

    def train_test_split(self, fraction: float = 0.25) -> None:
        """Split each client's data into train/test."""
        for client in self.clients:
            client.split_train_test(test_size=fraction)

        # Recompute weights based on training set size
        self.client_weights = np.array([client.get_weight() for client in self.clients])
        self.client_weights = self.client_weights / self.client_weights.sum()

    def sync_N(self) -> None:
        """Synchronize total sample count across clients."""
        N = sum(client.get_weight() for client in self.clients)
        for client in self.clients:
            client.set_N(N)

    def sync_Pa(self) -> None:
        """Synchronize P(A=0) across clients and compute alpha weights."""
        Pk0s = [client.get_Pka(a=0) for client in self.clients]
        Pa0 = (np.array(Pk0s) * self.client_weights).sum()
        for client in self.clients:
            client.set_alphaka(Pa0)

    def sample_clients(self) -> np.ndarray:
        """Sample clients for current round."""
        return np.random.choice(
            len(self.client_weights), self.m, p=self.client_weights, replace=False
        )

    def aggregate_theta(
        self, thetas: list[dict[str, torch.Tensor]], weights: list[float]
    ) -> None:
        """Aggregate client models into global model."""
        global_state_dict = {}
        for key in self.model.state_dict().keys():
            global_state_dict[key] = torch.zeros_like(
                self.model.state_dict()[key], device=self.device
            )

        # Weighted average
        for i, local_model in enumerate(thetas):
            for key in local_model.keys():
                global_state_dict[key] += local_model[key] * weights[i]

        self.model.load_state_dict(global_state_dict)

    def client_step(self) -> tuple[list[dict[str, torch.Tensor]], list[float]]:
        """Perform client steps for participating clients."""
        participating_client_ids = self.sample_clients()

        def copy_statedict(statedict):
            statedict_copy = {}
            for key in statedict.keys():
                statedict_copy[key] = torch.zeros_like(
                    statedict[key], device=self.device
                )
                statedict_copy[key] += statedict[key]
            return statedict_copy

        return (
            [
                self.clients[id].client_step(copy_statedict(self.model.state_dict()))
                for id in participating_client_ids
            ],
            [self.client_weights[id] for id in participating_client_ids],
        )

    def update_C(self) -> None:
        """Update prediction trackers (Algorithm 2 from Fair-FL)."""
        if self.Y_0 is None or self.Y_1 is None:
            return

        # Drop old predictions
        self.Y_0.drop(self.mu)
        self.Y_1.drop(self.mu)

        # Sample new predictions from clients
        updates0 = []
        updates1 = []

        def copy_statedict(statedict):
            statedict_copy = {}
            for key in statedict.keys():
                statedict_copy[key] = torch.zeros_like(
                    statedict[key], device=self.device
                )
                statedict_copy[key] += statedict[key]
            return statedict_copy

        for client, weight in zip(self.clients, self.client_weights):
            num_samples = [int(weight * self.mu * self.NY)] * 2
            p0, p1 = client.sample_C_update(
                num_samples, copy_statedict(self.model.state_dict())
            )
            updates0.append(p0)
            updates1.append(p1)

        self.Y_0.update(updates0)
        self.Y_1.update(updates1)

    def test_current_model(
        self,
    ) -> tuple[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], np.ndarray]:
        """Test current global model on all clients."""

        def copy_statedict(statedict):
            statedict_copy = {}
            for key in statedict.keys():
                statedict_copy[key] = torch.zeros_like(
                    statedict[key], device=self.device
                )
                statedict_copy[key] += statedict[key]
            return statedict_copy

        client_predictions = [
            client.test_client(copy_statedict(self.model.state_dict()))
            for client in self.clients
        ]
        return client_predictions, self.client_weights

    def log_progress(self) -> None:
        """Log accuracy and fairness metrics to wandb."""
        res, weights = self.test_current_model()

        # Concatenate all predictions
        all_preds = torch.cat([r[0] for r in res]).flatten()
        all_labels = torch.cat([r[1] for r in res]).flatten()
        all_sensitive = torch.cat([r[2] for r in res]).flatten()

        acc = accuracy(all_preds, all_labels)
        fairness = P1(all_preds, all_sensitive)

        wandb.log({"acc": acc, "fairness": fairness})

    def train(self) -> None:
        """Train federated learning model."""
        # Initialize prediction trackers
        self.Y_0 = PredictionTracker(demographic_group=0, capacity=self.NY)
        self.Y_1 = PredictionTracker(demographic_group=1, capacity=self.NY)

        # Sync Pa
        self.sync_Pa()

        # Training loop
        from tqdm import tqdm

        for t in tqdm(range(self.T)):
            # Update C function (Algorithm 2)
            self.update_C()

            # Set C on all clients
            for client in self.clients:
                client.set_C(self.Y_0, self.Y_1)

            # Client updates (Algorithm 3)
            model_updates, update_weights = self.client_step()
            self.aggregate_theta(model_updates, update_weights)

            # Log progress
            self.log_progress()

        wandb.finish()


def main():
    """Run Fair-FL experiment."""
    parser = argparse.ArgumentParser(description="Fair-FL Experiment Runner")
    parser.add_argument("--method", choices=["ours"], default="ours")
    parser.add_argument(
        "--home", required=True, help="Path to Fair-FL root directory (for datasets)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Directory to save .npy results (default: <home>/results/<method>)",
    )
    parser.add_argument("--numSeeds", default=10, type=int)
    parser.add_argument("--numComRnds", default=100, type=int)
    parser.add_argument("--numLambdas", default=50, type=int)
    parser.add_argument("--runName", default="run")
    parser.add_argument("--dataset", choices=["compas"], default="compas")

    args = parser.parse_args()

    HOMEFOLDER = args.home
    DEVICE = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {DEVICE}")

    # Lambda sweep
    ls = np.logspace(-5, 1, args.numLambdas)
    NYs = [100]  # COMPAS uses NY=100

    for NY in NYs:
        for lambda_val in ls:
            for seed in range(args.numSeeds):
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Load dataset
                if args.dataset == "compas":
                    datasets = CompasDataset().load_data(homefolder=HOMEFOLDER)
                    input_size = 8  # COMPAS has 8 features after one-hot encoding

                    def model_factory():
                        return TwoLayerNN(input_size)

                else:
                    raise ValueError(f"Unknown dataset: {args.dataset}")

                # Create server
                s = FairFLServer(
                    datasets,
                    model_factory,
                    torch.nn.BCEWithLogitsLoss(),
                    m=None,
                    T=args.numComRnds,
                    mu=1.0,
                    NY=NY,
                    lambda_=lambda_val,
                    datasetname=args.dataset.capitalize(),
                    runname=args.runName,
                    device=str(DEVICE),
                )

                # Train
                s.train_test_split()
                s.sync_N()
                s.sync_Pa()
                s.train()

                # Test
                res, weights = s.test_current_model()
                performances = np.vstack(
                    [np.array([accuracy(p, y), P1(p, a)]) for p, y, a in res]
                )

                # Save results
                output_dir = args.output or os.path.join(
                    HOMEFOLDER, f"results/{args.method}"
                )
                os.makedirs(output_dir, exist_ok=True)
                np.save(
                    os.path.join(
                        output_dir,
                        f"{args.dataset}_p_{lambda_val}_{seed}_{NY}.npy",
                    ),
                    performances,
                )


if __name__ == "__main__":
    main()
