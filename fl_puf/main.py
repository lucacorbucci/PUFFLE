import argparse
from typing import Dict

import flwr as fl
import wandb
from flwr.common.typing import Scalar

from fl_puf.Client.client import FlowerClient
from fl_puf.Utils.dataset_utils import DatasetDownloader
from fl_puf.Utils.model_utils import Learning
from fl_puf.Utils.utils import Utils

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument("--num_client_cpus", type=int, default=1)
parser.add_argument("--num_client_gpus", type=int, default=1)
parser.add_argument("--num_rounds", type=int, default=5)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--pool_size", type=int, default=100)
parser.add_argument("--sampled_clients", type=float, default=0.1)
parser.add_argument("--wandb", type=bool, default=False)
parser.add_argument("--DPL", type=bool, default=False)
parser.add_argument("--DPL_lambda", type=float, default=0.0)
parser.add_argument("--private", type=bool, default=False)
parser.add_argument("--epsilon", type=float, default=10)
parser.add_argument("--clipping", type=float, default=10)
parser.add_argument("--delta", type=float, default="0.1")
parser.add_argument("--lr", type=float, default="0.1")


# DPL:
# 1) baseline without privacy and DPL -> compute maximum violation
# 2) baseline without privacy and with DPL -> compute maximum violation
# 3) baseline with privacy and without DPL -> compute maximum violation
# 4) baseline with privacy and with DPL -> compute maximum violation


def setup_wandb(args):
    wandb_run = wandb.init(
        # set the wandb project where this run will be logged
        project="FL_fairness",
        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.02,
            "dataset": args.dataset,
            "num_rounds": args.num_rounds,
            "pool_size": args.pool_size,
            "sampled_clients": args.sampled_clients,
            "epochs": args.epochs,
        },
    )
    return wandb_run


def fit_config(server_round: int = 0) -> Dict[str, Scalar]:
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
    client_resources = {
        "num_cpus": args.num_client_cpus,
        "num_gpus": args.num_client_gpus,
    }

    if args.wandb:
        wandb_run = setup_wandb(args)
    else:
        wandb_run = None
    # Download CIFAR-10 dataset
    train_path, testset = DatasetDownloader.download_dataset(dataset_name)

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    fed_dir = DatasetDownloader.do_fl_partitioning(
        train_path,
        pool_size=pool_size,
        alpha=float("inf"),
        num_classes=2,
        val_ratio=0.2,
    )
    print(pool_size)

    # configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=args.sampled_clients,
        fraction_evaluate=args.sampled_clients,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=args.sampled_clients,  # All clients should be available
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
        evaluate_fn=Learning.get_evaluate_fn(
            testset, dataset_name, wandb_run, fit_config
        ),  # centralised evaluation of global model
    )

    def client_fn(cid: str):
        # create a single client instance
        return FlowerClient(
            cid,
            fed_dir,
            dataset_name,
            DPL=args.DPL,
            DPL_lambda=args.DPL_lambda,
            private=args.private,
            epsilon=args.epsilon,
            clipping=args.clipping,
            epochs=args.epochs,
            delta=args.delta,
            lr=args.lr,
        )

    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False}

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(
            num_rounds=args.num_rounds,
        ),
        strategy=strategy,
        ray_init_args=ray_init_args,
    )
    if wandb_run:
        wandb_run.finish()
