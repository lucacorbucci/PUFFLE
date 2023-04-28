import argparse
from typing import Dict

import flwr as fl
from flwr.common.typing import Scalar

from client import FlowerClient
from dataset_utils import DatasetDownloader
from utils import Utils


parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument("--num_client_cpus", type=int, default=1)
parser.add_argument("--num_rounds", type=int, default=5)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--pool_size", type=int, default=100)
parser.add_argument("--sampled_clients", type=float, default=0.1)


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": args.epochs,  # number of local epochs
        "batch_size": args.batch_size,
        "dataset": args.dataset,
    }
    return config


if __name__ == "__main__":
    # parse input arguments
    args = parser.parse_args()
    dataset_name = args.dataset

    pool_size = args.pool_size
    client_resources = {"num_cpus": args.num_client_cpus}

    # Download CIFAR-10 dataset
    train_path, testset = DatasetDownloader.download_dataset(dataset_name)

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    fed_dir = DatasetDownloader.do_fl_partitioning(
        train_path, pool_size=pool_size, alpha=1000, num_classes=10, val_ratio=0.1
    )

    # configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.2,
        fraction_evaluate=0.1,
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        evaluate_fn=Utils.get_evaluate_fn(
            testset,
            dataset_name,
        ),  # centralised evaluation of global model
    )

    def client_fn(cid: str):
        # create a single client instance
        return FlowerClient(cid, fed_dir, dataset_name)

    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )
