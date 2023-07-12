import argparse
import logging
import os
import random
import warnings
from typing import Dict

import flwr as fl
import numpy as np
import torch
import wandb
from DPL.Utils.dataset_utils import DatasetUtils
from DPL.Utils.model_utils import ModelUtils
from DPL.Utils.train_parameters import TrainParameters
from flwr.common.typing import Scalar
from flwr.server.client_manager import SimpleClientManager
from Server.server import Server
from Strategy.fed_avg import FedAvg
from torch import nn

from fl_puf.Client.client import FlowerClient
from fl_puf.Utils.utils import Utils

warnings.filterwarnings("ignore")

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
parser.add_argument("--epsilon", type=float, default=None)
parser.add_argument("--noise_multiplier", type=float, default=None)
parser.add_argument("--clipping", type=float, default=1000000000)
parser.add_argument("--delta", type=float, default=None)
parser.add_argument("--lr", type=float, default="0.1")
parser.add_argument("--alpha", type=int, default=1000000)
parser.add_argument("--train_csv", type=str, default="")
parser.add_argument("--test_csv", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--base_path", type=str, default="")

# DPL:
# 1) baseline without privacy and DPL -> compute maximum violation
# 2) baseline without privacy and with DPL -> compute maximum violation
# 3) baseline with privacy and without DPL -> compute maximum violation
# 4) baseline with privacy and with DPL -> compute maximum violation


def setup_wandb(args):
    wandb_run = wandb.init(
        # set the wandb project where this run will be logged
        project="FL_fairness",
        name=f"FL - Lambda {args.DPL_lambda} - LR {args.lr} - Batch {args.batch_size}",
        # track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "csv": args.train_csv,
            "DPL_regularization": args.DPL,
            "DPL_lambda": args.DPL_lambda,
            "batch_size": args.batch_size,
            "dataset": args.dataset,
            "num_rounds": args.num_rounds,
            "pool_size": args.pool_size,
            "sampled_clients": args.sampled_clients,
            "epochs": args.epochs,
            "private": args.private,
            "epsilon": args.epsilon if args.private else 0,
            "gradnorm": args.clipping,
            "delta": args.delta if args.private else 0,
            "noise_multiplier": args.noise_multiplier if args.private else 0,
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

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(args.seed)

    pool_size = args.pool_size
    client_resources = {
        "num_cpus": args.num_client_cpus,
        "num_gpus": args.num_client_gpus,
    }

    wandb_run = setup_wandb(args) if args.wandb else None
    # Download CIFAR-10 dataset
    train_set, test_set = DatasetUtils.download_dataset(
        dataset_name,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        debug=args.debug,
    )
    train_path = Utils.prepare_dataset_for_FL(
        train_set=train_set,
        dataset_name=dataset_name,
        base_path=args.base_path,
    )
    train_parameters = TrainParameters(
        epochs=args.epochs,
        device="cuda" if torch.cuda.is_available() else "cpu",
        criterion=nn.CrossEntropyLoss(),
        wandb_run=None,
        batch_size=args.batch_size,
        seed=args.seed,
        epsilon=args.epsilon if args.private else None,
        DPL_lambda=args.DPL_lambda,
        private=args.private,
        DPL=args.DPL,
        noise_multiplier=args.noise_multiplier,
    )

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    fed_dir = Utils.do_fl_partitioning(
        train_path,
        pool_size=pool_size,
        num_classes=2,
        val_ratio=0,
        partition_type="iid",
    )

    test = os.listdir(fed_dir)

    for item in test:
        if item.endswith(".pkl"):
            os.remove(os.path.join(fed_dir, item))

    def client_fn(cid: str):
        # create a single client instance

        return FlowerClient(
            train_parameters=train_parameters,
            cid=cid,
            fed_dir_data=fed_dir,
            dataset_name=dataset_name,
            clipping=args.clipping,
            delta=args.delta,
            lr=args.lr,
        )

    model = ModelUtils.get_model(dataset_name, "cuda")
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(model_parameters)

    def agg_metrics_train(metrics: list, server_round: int) -> dict:
        # Collect all the FL Client metrics and weight them
        losses = [n_examples * metric["train_loss"] for n_examples, metric in metrics]
        losses_with_regularization = [
            n_examples * metric["train_loss_with_regularization"]
            for n_examples, metric in metrics
        ]
        epsilon_list = [metric["epsilon"] for _, metric in metrics]

        accuracies = [
            n_examples * metric["train_accuracy"] for n_examples, metric in metrics
        ]

        total_examples = sum([n_examples for n_examples, _ in metrics])

        # Compute weighted averages
        agg_metrics = {
            "train_loss": sum(losses) / total_examples,
            "train_accuracy": sum(accuracies) / total_examples,
            "train_loss_with_regularization": sum(losses_with_regularization)
            / total_examples,
        }

        wandb_run.log(
            {
                "Train Loss": agg_metrics["train_loss"],
                "Train Accuracy": agg_metrics["train_accuracy"],
                "Train Loss with Regularization": agg_metrics[
                    "train_loss_with_regularization"
                ],
                "Train Epsilon": max(epsilon_list),
                "FL Round": server_round,
            }
        )

        return agg_metrics

    strategy = FedAvg(
        fraction_fit=args.sampled_clients,
        fraction_evaluate=0,
        min_fit_clients=args.sampled_clients,
        min_evaluate_clients=0,
        min_available_clients=args.sampled_clients,
        on_fit_config_fn=fit_config,
        evaluate_fn=Utils.get_evaluate_fn(
            test_set=test_set,
            dataset_name=dataset_name,
            train_parameters=train_parameters,
            wandb_run=wandb_run,
            batch_size=args.batch_size,
        ),  # centralised evaluation of global model
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=agg_metrics_train,
    )

    ray_num_cpus = 2
    ray_num_gpus = 3
    ram_memory = 16_000 * 1024 * 1024 * 2

    # (optional) specify Ray config
    ray_init_args = {
        "include_dashboard": False,
        "num_cpus": ray_num_cpus,
        "num_gpus": ray_num_gpus,
        "_memory": ram_memory,
        "_redis_max_memory": 100000000,
        "object_store_memory": 100000000,
        "logging_level": logging.ERROR,
        "log_to_driver": True,
    }

    client_manager = SimpleClientManager()
    server = Server(client_manager=client_manager, strategy=strategy)

    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
        server=server,
        client_manager=client_manager,
    )

    if wandb_run:
        wandb_run.finish()
