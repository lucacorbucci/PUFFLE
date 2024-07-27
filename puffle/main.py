"""
Implementation of the paper "PUFFLE: Balancing Privacy, Utility, and Fairness in
Federated Learning" by L. Corbucci, M. A. Heikkilä, D.S. Noguero, A. Monreale, N. Kourtellis.

https://arxiv.org/abs/2407.15224

Code developed by L. Corbucci, 2024
"""

import argparse
import logging
import os
import signal
import sys
import time
import warnings
from logging import INFO
from typing import Dict

import flwr as fl
import numpy as np
import torch
from Utils.aggregations_fuctions import AggregationFunctions
from Utils.dataset_utils import DatasetUtils
from Utils.model_utils import ModelUtils
from Utils.tabular_data_loader import prepare_tabular_data
from Utils.train_parameters import TrainParameters
from Utils.utils import Utils
from Client.client import FlowerClientDisparity
from ClientManager.client_manager import SimpleClientManager
from Strategy.fed_avg import FedAvg
from flwr.common.logger import log
from flwr.common.typing import Scalar
from Server.server import Server


def signal_handler(sig, frame):
    print("Gracefully stopping your experiment! Keep calm!")
    global wandb_run
    if wandb_run:
        wandb_run.finish()
    sys.exit(0)


warnings.filterwarnings("ignore")

# ----------------------------- Parsing Options ----------------------------------------
parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

# ----------------------
# Experiment Settings
parser.add_argument(
    "--num_client_cpus", type=float, default=1
)  # Percentage of CPUs used by each client
parser.add_argument(
    "--num_client_gpus", type=float, default=1
)  # Percentage of GPUs used by each client
parser.add_argument(
    "--num_rounds", type=int, default=5
)  # Number of rounds of federated learning
# Dataset we want to use, based on this we will use a different model
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--pool_size", type=int, default=100)  # Number of clients
parser.add_argument(
    "--wandb", type=bool, default=False
)  # If we want to use wandb to log the results
parser.add_argument(
    "--cross_silo", type=bool, default=False
)  # If we are in a cross silo scenario
parser.add_argument(
    "--seed", type=int, default=42
)  # Used to set the seed of the random functions and to sample every time the same test clients
parser.add_argument(
    "--debug", type=bool, default=False
)  # Used to print debug information
parser.add_argument(
    "--probability_estimation", type=bool, default=False
)  # If we want to use the probability estimation
parser.add_argument("--perfect_probability_estimation", type=bool, default=False)


# ----------------------
# Privacy Settings
parser.add_argument(
    "--regularization", type=bool, default=False
)  # If we want to use DPL Regularization
# parser.add_argument("--private", type=bool, default=True)  # If we want to use DP-SGD
parser.add_argument("--epsilon", type=float, default=None)  # Target Epsilon for DP-SGD
parser.add_argument(
    "--epsilon_lambda", type=float, default=None
)  # Target Epsilon for Lambda computation
parser.add_argument(
    "--epsilon_statistics", type=float, default=None
)  # Target Epsilon for statistics sharing
# parser.add_argument(
#     "--noise_multiplier", type=float, default=None
# )  # Noise multiplier for DP-SGD
parser.add_argument(
    "--clipping", type=float, default=1000000000
)  # Clipping value for DP-SGD
# parser.add_argument("--delta", type=float, default=None)

# ----------------------
# Dataset/Distribution Settings
parser.add_argument("--train_csv", type=str, default="")
parser.add_argument("--base_path", type=str, default="")
parser.add_argument("--sort_clients", type=bool, default=True)
parser.add_argument("--no_sort_clients", dest="sort_clients", action="store_false")
parser.add_argument(
    "--node_shuffle_seed", type=int, default=None
)  # Seed to shuffle the nodes of validation/train group but not the test group

parser.add_argument(
    "--sweep", type=bool, default=False
)  # true if we are using wandb sweep to tune the hyperparameters

# ----------------------
# Hyperparameters
parser.add_argument(
    "--weight_decay_lambda", type=float, default=None
)  # weight decay for the alpha
parser.add_argument("--optimizer", type=str, default=0)  # optimizer we want to use
parser.add_argument(
    "--alpha_target_lambda", type=float, default=None
)  # alpha (velocity) to update the lambda
parser.add_argument("--epochs", type=int, default=1)  # Number of epochs per round
parser.add_argument("--batch_size", type=int, default=64)  # Batch size
parser.add_argument("--lr", type=float, default="0.1")

# ----------------------
# Percentage of samples sampled from train, validation and test group
parser.add_argument(
    "--sampled_clients", type=float, default=0.1
)  # Percentage of training clients sampled
parser.add_argument(
    "--sampled_clients_test", type=float, default=0.1
)  # Percentage of test clients sampled
parser.add_argument(
    "--sampled_clients_validation", type=float, default=0
)  # Percentage of validation clients sampled

# ----------------------
# Distributions of training/validation/test nodes
parser.add_argument(
    "--training_nodes", type=float, default=0
)  # Percentage of training nodes
parser.add_argument(
    "--validation_nodes", type=float, default=0
)  # Percentage of validation nodes
parser.add_argument("--test_nodes", type=float, default=0)  # Percentage of test nodes
parser.add_argument("--project_name", type=str, default=None)  # project name for wandb
parser.add_argument("--run_name", type=str, default=None)  # run name for wandb
parser.add_argument(
    "--splitted_data_dir", type=str, default="federated"
)  # change the name of the fed_dir to run multiple experiments in parallel


# ----------------------
# Parameters for the Lambda
parser.add_argument(
    "--regularization_mode", type=str, default=None
)  # This is either fixed get_tabular_dataor tunable based on if we want to update the Lambda
#  during the training or not
parser.add_argument(
    "--metric", type=str, default="disparity"
)  # This is either fixed or tunable based on if we want to update the Lambda
#  during the training or not
parser.add_argument(
    "--regularization_lambda", type=float, default=None
)  # value to initialize the Lambda, this is used when starting_lambda_mode is fixed
parser.add_argument(
    "--update_lambda", type=bool, default=False
)  # if we want to update the Lambda during the training

parser.add_argument(
    "--momentum", type=float, default=0
)  # Momentum value applied to the Lambda

parser.add_argument("--target", type=float, default=None)


# ----------------------
# Partitioning Settings for NON TABULAR DATA
parser.add_argument("--partition_type", type=str, default="non_iid")
parser.add_argument("--percentage_unbalanced_nodes", type=float, default=None)
parser.add_argument("--unbalanced_ratio", type=float, default=0.4)
parser.add_argument(
    "--alpha", type=int, default=1000000
)  # Alpha of the dirichlet distribution

# ----------------------
# Parameters to generate the tabular dataset with Mikko's implementation
parser.add_argument("--tabular_data", type=bool, default=False)
parser.add_argument("--dataset_path", type=str, default="../data/celeba")
parser.add_argument("--groups_balance_factor", type=float, default=0.9)
parser.add_argument("--priv_balance_factor", type=float, default=0.5)

# ----------------------
# Parameters to generate the tabular dataset
parser.add_argument(
    "--group_to_reduce", type=int, default=None, nargs="+"
)  # group of <target, sensitive value> that we want to reduce
parser.add_argument(
    "--group_to_increment", type=int, default=None, nargs="+"
)  # group of <target, sensitive value> that we want to increment
parser.add_argument(
    "--opposite_group_to_reduce", type=int, default=None, nargs="+"
)  # group of <target, sensitive value> that we want to reduce when opposite_direction is true
parser.add_argument(
    "--opposite_group_to_increment", type=int, default=None, nargs="+"
)  # group of <target, sensitive value> that we want to increment when opposite_direction is true
parser.add_argument(
    "--number_of_samples_per_node", type=int, default=None
)  # maximum number of samples per node
parser.add_argument(
    "--ratio_unfairness", type=float, default=None, nargs="+"
)  # percentage of samples removed from group_to_reduce on the unfair nodes
parser.add_argument(
    "--opposite_ratio_unfairness", type=float, default=None, nargs="+"
)  # percentage of samples removed from group_to_reduce one the unfair nodes when opposite_direction is true
parser.add_argument(
    "--ratio_unfair_nodes", type=float, default=None
)  # percentage of nodes that will be unbalanced (unfair nodes)
parser.add_argument(
    "--opposite_direction", type=bool, default=False
)  # If we want a disparity in some nodes that is opposite to the one in the others
parser.add_argument(
    "--approach", type=str, default=""
)  # The approach we want to use to generate the dataset, can be egalitarian or representative
parser.add_argument(
    "--strategy", type=str, default="fedavg"
)  # The approach we want to use to generate the dataset, can be egalitarian or representative

parser.add_argument(
    "--one_group_nodes", type=bool, default=False
)  # The approach we want to use to generate the dataset, can be egalitarian or representative


parser.add_argument(
    "--privileged_group", type=int, default=None, nargs="+"
)  # The approach we want to use to generate the dataset, can be egalitarian or representative
parser.add_argument(
    "--unprivileged_group", type=int, default=None, nargs="+"
)  # The approach we want to use to generate the dataset, can be egalitarian or representative

parser.add_argument("--store_model", type=bool, default=False)
parser.add_argument("--file_name", type=str, default=None)
parser.add_argument("--f1", type=bool, default=False)
parser.add_argument("--comparison", type=bool, default=False)
parser.add_argument("--switch_dataset", type=int, default=None)

# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    # remove files in tmp/ray
    args = parser.parse_args()
    dataset_name = args.dataset
    if args.node_shuffle_seed is None:
        node_shuffle_seed = int(str(time.time()).split(".")[1]) * args.seed
        args.node_shuffle_seed = node_shuffle_seed

    print(
        f"Removing files in {args.dataset_path}/celeba-10-batches-py/{args.splitted_data_dir}/*.pkl"
    )
    if args.dataset == "celeba":
        os.system(
            f"rm -rf {args.dataset_path}/celeba-10-batches-py/{args.splitted_data_dir}/*.pkl"
        )
    else:
        os.system(f"rm -rf {args.dataset_path}{args.splitted_data_dir}/*.pkl")

    Utils.seed_everything(args.seed)

    current_max_epsilon = 0
    pool_size = args.pool_size
    client_resources = {
        "num_cpus": args.num_client_cpus,
        "num_gpus": args.num_client_gpus,
    }

    DPL_value = None

    # We check if we pass parameters that are not compatible with each other
    if args.regularization_mode == "fixed" and args.regularization_lambda is None:
        raise Exception("Starting lambda value must be specified when using fixed mode")
    elif args.regularization_mode is None:
        DPL_value = 0
        args.regularization_lambda = 0
    elif args.regularization_mode == "fixed" and args.regularization_lambda:
        DPL_value = args.regularization_lambda
    elif args.regularization_mode != "fixed" and args.regularization_mode != "tunable":
        raise Exception(
            f"Starting lambda mode not recognized, your value is {args.regularization_mode}. You need to use either fixed or tunable "
        )

    print("The regularization_lambda is: ", args.regularization_lambda)
    print("The epsilon: ", args.epsilon)

    num_training_nodes = int(args.pool_size * args.training_nodes)
    num_validation_nodes = int(args.pool_size * args.validation_nodes)
    num_test_nodes = int(args.pool_size * args.test_nodes)

    train_parameters = TrainParameters(
        epochs=args.epochs,
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=args.batch_size,
        seed=args.seed,
        optimizer=args.optimizer,
        regularization=args.regularization,
        regularization_mode=args.regularization_mode,
        regularization_lambda=args.regularization_lambda,
        target=args.target,
        fl_round=args.num_rounds,
        weight_decay_alpha=args.weight_decay_lambda,
        momentum=args.momentum,
        sweep=args.sweep,
        alpha=args.alpha_target_lambda,
        epsilon=args.epsilon,
        epsilon_lambda=args.epsilon_lambda,
        epsilon_statistics=args.epsilon_statistics,
        metric=args.metric,
        privileged_group=(
            tuple(args.privileged_group) if args.privileged_group else None
        ),
        unprivileged_group=(
            tuple(args.unprivileged_group) if args.unprivileged_group else None
        ),
    )

    if args.tabular_data:
        # If we are using a tabular dataset we have a different way to load and
        # split the dataset into clients
        fed_dir, _ = prepare_tabular_data(
            dataset_path=args.dataset_path,
            dataset_name=dataset_name,
            do_iid_split=True,
            approach=args.approach,
            num_nodes=args.pool_size,
            ratio_unfair_nodes=args.ratio_unfair_nodes,
            opposite_direction=args.opposite_direction,
            ratio_unfairness=(
                tuple(args.ratio_unfairness) if args.ratio_unfairness else None
            ),
            group_to_reduce=(
                tuple(args.group_to_reduce) if args.group_to_reduce else None
            ),
            group_to_increment=(
                tuple(args.group_to_increment) if args.group_to_increment else None
            ),
            number_of_samples_per_node=args.number_of_samples_per_node,
            opposite_group_to_reduce=(
                tuple(args.opposite_group_to_reduce)
                if args.opposite_group_to_reduce
                else None
            ),
            opposite_group_to_increment=(
                tuple(args.opposite_group_to_increment)
                if args.opposite_group_to_increment
                else None
            ),
            opposite_ratio_unfairness=(
                tuple(args.opposite_ratio_unfairness)
                if args.opposite_ratio_unfairness
                else None
            ),
            one_group_nodes=args.one_group_nodes,
            splitted_data_dir=args.splitted_data_dir,
        )
    else:
        # If we are not using a tabular dataset we have a different way to load and
        # split the dataset into clients
        train_set, test_set = DatasetUtils.load_dataset(
            dataset_name,
            train_csv=args.train_csv,
            debug=False,
            base_path=args.dataset_path,
        )

        train_path = Utils.prepare_dataset_for_FL(
            dataset=train_set,
            dataset_name=dataset_name,
            base_path=args.base_path,
        )
        # Partitioning the training dataset
        fed_dir = Utils.do_fl_partitioning(
            train_path,
            pool_size=pool_size,
            num_classes=2,
            val_ratio=0,
            partition_type=args.partition_type,
            alpha=args.alpha,
            train_parameters=train_parameters,
            group_to_reduce=(
                tuple(args.group_to_reduce) if args.group_to_reduce else None
            ),
            group_to_increment=(
                tuple(args.group_to_increment) if args.group_to_increment else None
            ),
            number_of_samples_per_node=args.number_of_samples_per_node,
            ratio_unfair_nodes=(
                args.ratio_unfair_nodes if args.ratio_unfair_nodes else None
            ),
            ratio_unfairness=(
                tuple(args.ratio_unfairness) if args.ratio_unfairness else None
            ),
            one_group_nodes=args.one_group_nodes,
            splitted_data_dir=args.splitted_data_dir,
        )
        path_to_remove = os.listdir(fed_dir)
        for item in path_to_remove:
            if item.endswith(".pkl"):
                os.remove(os.path.join(fed_dir, item))

    wandb_run = Utils.setup_wandb(args, train_parameters) if args.wandb else None

    def client_fn(cid: str):
        client_generator = np.random.default_rng(seed=[args.seed, cid])
        # create a single client instance
        if args.metric == "disparity":
            return FlowerClientDisparity(
                train_parameters=train_parameters,
                cid=cid,
                fed_dir_data=fed_dir,
                dataset_name=dataset_name,
                clipping=args.clipping,
                lr=args.lr,
                client_generator=client_generator,
            )
        else:
            raise Exception("Metric not recognized")

    model = ModelUtils.get_model(dataset_name, "cuda")
    model = model.to("cuda")
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    initial_parameters = fl.common.ndarrays_to_parameters(model_parameters)

    def fit_config(server_round: int = 0) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epochs": args.epochs,  # number of local epochs
            "batch_size": args.batch_size,
            "dataset": args.dataset,
            "server_round": server_round,
        }
        return config

    def evaluate_config(server_round: int = 0) -> Dict[str, Scalar]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "epochs": args.epochs,  # number of local epochs
            "batch_size": args.batch_size,
            "dataset": args.dataset,
        }
        return config

    log(
        INFO,
        f"CLIENT SAMPLED: {args.sampled_clients}, {args.sampled_clients_validation}, {args.sampled_clients_test}",
    )

    strategy = FedAvg(
        fraction_fit=args.sampled_clients,
        fraction_evaluate=args.sampled_clients_validation,
        fraction_test=args.sampled_clients_test,
        min_fit_clients=args.sampled_clients,
        min_evaluate_clients=0,
        min_available_clients=args.sampled_clients,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=AggregationFunctions.agg_metrics_train,
        evaluate_metrics_aggregation_fn=AggregationFunctions.agg_metrics_evaluation,
        test_metrics_aggregation_fn=AggregationFunctions.agg_metrics_test,
        current_max_epsilon=current_max_epsilon,
        fed_dir=fed_dir,
        model=ModelUtils.get_model(dataset_name, "cuda"),
        file_name=args.file_name,
        store_model=args.store_model,
        wandb=wandb_run,
        args=args,
        train_parameters=train_parameters,
    )

    # these parameters are used to configure Ray and they are dependent on
    # the machine we want to use to run the experiments
    ray_num_cpus = 20
    ray_num_gpus = 1
    ram_memory = 16_000 * 1024 * 1024 * 2

    # (optional) specify Ray config
    ray_init_args = {
        "include_dashboard": False,
        "num_cpus": ray_num_cpus,
        "num_gpus": ray_num_gpus,
        "_memory": ram_memory,
        "_redis_max_memory": 10000000,
        "object_store_memory": 78643200,
        "logging_level": logging.ERROR,
        "log_to_driver": True,
    }

    client_manager = SimpleClientManager(
        seed=args.seed,
        num_clients=pool_size,
        sort_clients=args.sort_clients,
        num_training_nodes=num_training_nodes,
        num_validation_nodes=num_validation_nodes,
        num_test_nodes=num_test_nodes,
        node_shuffle_seed=(
            node_shuffle_seed
            if args.node_shuffle_seed is None
            else args.node_shuffle_seed
        ),
        fed_dir=fed_dir,
        ratio_unfair_nodes=args.ratio_unfair_nodes,
        fl_rounds=args.num_rounds,
        fraction_fit=args.sampled_clients,
        fraction_evaluate=args.sampled_clients_validation,
        fraction_test=args.sampled_clients_test,
    )
    server = Server(client_manager=client_manager, strategy=strategy, args=args)

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
