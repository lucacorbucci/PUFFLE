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
from ClientManager.client_manager import SimpleClientManager
from DPL.Utils.dataset_utils import DatasetUtils
from DPL.Utils.model_utils import ModelUtils
from flwr.common.typing import Scalar
from opacus import PrivacyEngine
from Server.server import Server
from Strategy.fed_avg import FedAvg
from torch import nn
from Utils.train_parameters import TrainParameters

from fl_puf.Client.client import FlowerClient
from fl_puf.Utils.utils import Utils

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument("--num_client_cpus", type=float, default=1)
parser.add_argument("--num_client_gpus", type=float, default=1)
parser.add_argument("--num_rounds", type=int, default=5)
parser.add_argument("--dataset", type=str, default=None)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--pool_size", type=int, default=100)
parser.add_argument("--sampled_clients", type=float, default=0.1)
parser.add_argument("--sampled_clients_test", type=float, default=0.1)
parser.add_argument("--sampled_clients_validation", type=float, default=0)
parser.add_argument("--wandb", type=bool, default=False)
parser.add_argument("--DPL", type=bool, default=False)
parser.add_argument("--private", type=bool, default=False)
parser.add_argument("--epsilon", type=float, default=None)
parser.add_argument("--noise_multiplier", type=float, default=0)
parser.add_argument("--clipping", type=float, default=1000000000)
parser.add_argument("--delta", type=float, default=None)
parser.add_argument("--lr", type=float, default="0.1")
parser.add_argument("--alpha", type=int, default=1000000)
parser.add_argument("--train_csv", type=str, default="")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--debug", type=bool, default=False)
parser.add_argument("--base_path", type=str, default="")
parser.add_argument("--probability_estimation", type=bool, default=False)
parser.add_argument("--perfect_probability_estimation", type=bool, default=False)
parser.add_argument("--partition_type", type=str, default="non_iid")
parser.add_argument("--partition_ratio", type=float, default=None)
parser.add_argument("--sort_clients", type=bool, default=True)
parser.add_argument("--no_sort_clients", dest="sort_clients", action="store_false")
parser.add_argument("--alpha_target_lambda", type=float, default=None)
parser.add_argument("--target", type=float, default=None)
parser.add_argument("--cross_silo", type=bool, default=False)
parser.add_argument("--weight_decay_lambda", type=float, default=None)
parser.add_argument("--sweep", type=bool, default=False)
parser.add_argument("--optimizer", type=str, default=0)
parser.add_argument("--training_nodes", type=float, default=0)
parser.add_argument("--validation_nodes", type=float, default=0)
parser.add_argument("--test_nodes", type=float, default=0)
parser.add_argument("--node_shuffle_seed", type=int, default=30)
parser.add_argument("--starting_lambda_mode", type=str, default=None)
parser.add_argument("--starting_lambda_value", type=float, default=None)
parser.add_argument("--momentum", type=float, default=None)


# DPL:
# 1) baseline without privacy and DPL -> compute maximum violation
# 2) baseline without privacy and with DPL -> compute maximum violation
# 3) baseline with privacy and without DPL -> compute maximum violation
# 4) baseline with privacy and with DPL -> compute maximum violation
def get_optimizer(model, train_parameters, lr):
    if train_parameters.optimizer == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )
    elif train_parameters.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
        )
    elif train_parameters.optimizer == "adamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
        )
    else:
        raise ValueError("Optimizer not recognized")


def setup_wandb(args, train_parameters):
    if train_parameters.noise_multiplier > 0:
        noise_multiplier = train_parameters.noise_multiplier
    elif args.noise_multiplier > 0:
        noise_multiplier = args.noise_multiplier
    else:
        noise_multiplier = 0
    wandb_run = wandb.init(
        # set the wandb project where this run will be logged
        project="FL_fairness",
        # name=f"FL - Lambda {args.DPL_lambda} - LR {args.lr} - Batch {args.batch_size}",
        # track hyperparameters and run metadata
        config={
            "learning_rate": args.lr,
            "csv": args.train_csv,
            "DPL_regularization": args.DPL,
            # "DPL_lambda": args.DPL_lambda,
            "batch_size": args.batch_size,
            "dataset": args.dataset,
            "num_rounds": args.num_rounds,
            "pool_size": args.pool_size,
            "sampled_clients": args.sampled_clients,
            "epochs": args.epochs,
            "private": args.private,
            "epsilon": args.epsilon if args.private else None,
            "gradnorm": args.clipping,
            "delta": args.delta if args.private else 0,
            "noise_multiplier": noise_multiplier,
            "probability_estimation": args.probability_estimation,
            "perfect_probability_estimation": args.perfect_probability_estimation,
            "alpha": args.alpha,
            "partition_ratio": args.partition_ratio,
            "alpha_target_lambda": args.alpha_target_lambda,
            "target": args.target,
            "weight_decay_lambda": args.weight_decay_lambda,
            "starting_lambda_mode": args.starting_lambda_mode,
            "starting_lambda_value": args.starting_lambda_value,
            "momentum": args.momentum,
            "node_shuffle_seed": args.node_shuffle_seed,
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


def evaluate_config(server_round: int = 0) -> Dict[str, Scalar]:
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

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    current_max_epsilon = 0
    pool_size = args.pool_size
    client_resources = {
        "num_cpus": args.num_client_cpus,
        "num_gpus": args.num_client_gpus,
    }

    train_set, test_set = DatasetUtils.download_dataset(
        dataset_name,
        train_csv=args.train_csv,
        debug=args.debug,
    )
    train_path = Utils.prepare_dataset_for_FL(
        dataset=train_set,
        dataset_name=dataset_name,
        base_path=args.base_path,
    )

    DPL_value = None
    if args.starting_lambda_mode == "fixed" and args.starting_lambda_value is None:
        raise Exception("Starting lambda value must be specified when using fixed mode")
    elif args.starting_lambda_mode == "fixed" and args.starting_lambda_value:
        DPL_value = args.starting_lambda_value

    if (
        args.starting_lambda_mode != "fixed"
        and args.starting_lambda_mode != "avg"
        and args.starting_lambda_mode != "disparity"
        and args.starting_lambda_mode != "no_tuning"
    ):
        raise Exception(
            f"Starting lambda mode not recognized, your value is {args.starting_lambda_mode}"
        )
    train_parameters = TrainParameters(
        epochs=args.epochs,
        device="cuda" if torch.cuda.is_available() else "cpu",
        criterion=nn.CrossEntropyLoss(),
        wandb_run=None,
        batch_size=args.batch_size,
        seed=args.seed,
        epsilon=args.epsilon if args.private else None,
        DPL_lambda=DPL_value,
        private=args.private,
        DPL=args.DPL,
        noise_multiplier=args.noise_multiplier,
        probability_estimation=args.probability_estimation,
        perfect_probability_estimation=args.perfect_probability_estimation,
        partition_ratio=args.partition_ratio,
        target=args.target,
        alpha=args.alpha_target_lambda,
        cross_silo=args.cross_silo,
        weight_decay_lambda=args.weight_decay_lambda,
        sweep=args.sweep,
        starting_lambda_mode=args.starting_lambda_mode,
        starting_lambda_value=args.starting_lambda_value,
        momentum=args.momentum,
    )

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.

    # Partitioning the training dataset
    fed_dir = Utils.do_fl_partitioning(
        train_path,
        pool_size=pool_size,
        num_classes=2,
        val_ratio=0,
        partition_type=args.partition_type,
        alpha=args.alpha,
        train_parameters=train_parameters,
    )

    print(fed_dir)
    test = os.listdir(fed_dir)

    for item in test:
        if item.endswith(".pkl"):
            os.remove(os.path.join(fed_dir, item))

    if args.epsilon:
        # We need to understand the noise that we need to add based
        # on the epsilon that we want to guarantee
        max_noise = 0
        for i in range(args.pool_size):
            model_noise = ModelUtils.get_model(
                dataset_name, device=train_parameters.device
            )
            # get the training dataset of one of the clients
            train_loader_client_0 = Utils.get_dataloader(
                fed_dir,
                str(i),
                batch_size=train_parameters.batch_size,
                workers=0,
                dataset=dataset_name,
                partition="train",
            )
            privacy_engine = PrivacyEngine(accountant="rdp")
            optimizer_noise = get_optimizer(model_noise, train_parameters, args.lr)
            (
                _,
                private_optimizer,
                _,
            ) = privacy_engine.make_private_with_epsilon(
                module=model_noise,
                optimizer=optimizer_noise,
                data_loader=train_loader_client_0,
                epochs=(args.num_rounds // 10) * args.epochs,
                target_epsilon=train_parameters.epsilon,
                target_delta=args.delta,
                max_grad_norm=args.clipping,
            )
            max_noise = max(max_noise, private_optimizer.noise_multiplier)
            print(
                f"Node {i} - {(args.num_rounds // 10) * args.epochs} -- {private_optimizer.noise_multiplier}"
            )

        train_parameters.noise_multiplier = max_noise
        train_parameters.epsilon = None
        print(
            f">>>>> FINALE {(args.num_rounds // 10) * args.epochs} -- {train_parameters.noise_multiplier}"
        )

    wandb_run = setup_wandb(args, train_parameters) if args.wandb else None

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

    def agg_metrics_test(metrics: list, server_round: int) -> dict:
        total_examples = sum([n_examples for n_examples, _ in metrics])

        loss_test = (
            sum(
                [
                    n_examples
                    * metric[
                        "test_loss" if not train_parameters.sweep else "validation_loss"
                    ]
                    for n_examples, metric in metrics
                ]
            )
            / total_examples
        )
        accuracy_test = (
            sum(
                [
                    n_examples
                    * metric[
                        "test_accuracy"
                        if not train_parameters.sweep
                        else "validation_accuracy"
                    ]
                    for n_examples, metric in metrics
                ]
            )
            / total_examples
        )
        max_disparity_average = np.mean(
            [
                metric[
                    "max_disparity_test"
                    if not train_parameters.sweep
                    else "max_disparity_validation"
                ]
                for n_examples, metric in metrics
            ]
        )

        combinations = ["0|0", "0|1", "1|0", "1|1"]
        targets = ["0", "1"]

        sum_counters = {"0|0": 0, "0|1": 0, "1|0": 0, "1|1": 0}
        sum_targets = {"0": 0, "1": 0}

        for _, metric in metrics:
            metric = metric["counters"]
            for combination in combinations:
                sum_counters[combination] += metric[combination]
            for target in targets:
                sum_targets[target] += metric[target]
        max_disparity_statistics = max(
            [
                sum_counters["0|0"] / sum_targets["0"]
                - sum_counters["0|1"] / sum_targets["1"],
                sum_counters["0|1"] / sum_targets["1"]
                - sum_counters["0|0"] / sum_targets["0"],
                sum_counters["1|0"] / sum_targets["0"]
                - sum_counters["1|1"] / sum_targets["1"],
                sum_counters["1|1"] / sum_targets["1"]
                - sum_counters["1|0"] / sum_targets["0"],
            ],
        )

        agg_metrics = {
            "Test Loss": loss_test,
            "Test Accuracy": accuracy_test,
            "Test Disparity with average": max_disparity_average,
            "Test Disparity with statistics": max_disparity_statistics,
            "FL Round": server_round,
        }

        if wandb_run:
            wandb_run.log(agg_metrics)
        return agg_metrics

    def agg_metrics_evaluation(metrics: list, server_round: int) -> dict:
        total_examples = sum([n_examples for n_examples, _ in metrics])

        loss_evaluation = (
            sum(
                [
                    n_examples
                    * metric[
                        "test_loss" if not train_parameters.sweep else "validation_loss"
                    ]
                    for n_examples, metric in metrics
                ]
            )
            / total_examples
        )
        accuracy_evaluation = (
            sum(
                [
                    n_examples
                    * metric[
                        "test_accuracy"
                        if not train_parameters.sweep
                        else "validation_accuracy"
                    ]
                    for n_examples, metric in metrics
                ]
            )
            / total_examples
        )
        max_disparity_average = np.mean(
            [
                metric[
                    "max_disparity_test"
                    if not train_parameters.sweep
                    else "max_disparity_validation"
                ]
                for n_examples, metric in metrics
            ]
        )

        combinations = ["0|0", "0|1", "1|0", "1|1"]
        targets = ["0", "1"]

        sum_counters = {"0|0": 0, "0|1": 0, "1|0": 0, "1|1": 0}
        sum_targets = {"0": 0, "1": 0}

        for _, metric in metrics:
            metric = metric["counters"]
            for combination in combinations:
                sum_counters[combination] += metric[combination]
            for target in targets:
                sum_targets[target] += metric[target]
        max_disparity_statistics = max(
            [
                sum_counters["0|0"] / sum_targets["0"]
                - sum_counters["0|1"] / sum_targets["1"],
                sum_counters["0|1"] / sum_targets["1"]
                - sum_counters["0|0"] / sum_targets["0"],
                sum_counters["1|0"] / sum_targets["0"]
                - sum_counters["1|1"] / sum_targets["1"],
                sum_counters["1|1"] / sum_targets["1"]
                - sum_counters["1|0"] / sum_targets["0"],
            ]
        )

        custom_metric = accuracy_evaluation
        if args.target:
            distance = args.target - max_disparity_statistics
            distance = 0 if distance >= 0 else distance

            # custom_metric will be -inf when the disparity is above the target
            # otherwise we will have a positive value that depends on the distance
            # and on the accuracy on the validation set
            custom_metric = accuracy_evaluation + distance

        agg_metrics = {
            "Validation Loss": loss_evaluation,
            "Validation_Accuracy": accuracy_evaluation,
            "Validation Disparity with average": max_disparity_average,
            "Validation Disparity with statistics": max_disparity_statistics,
            "Custom_metric": custom_metric,
            "FL Round": server_round,
        }

        if wandb_run:
            wandb_run.log(agg_metrics)
        return agg_metrics

    def agg_metrics_train(
        metrics: list, server_round: int, current_max_epsilon: float, fed_dir
    ) -> dict:
        # Collect the losses logged during each epoch in each client
        total_examples = sum([n_examples for n_examples, _ in metrics])

        losses = []
        losses_with_regularization = []
        epsilon_list = []
        accuracies = []
        lambda_list = []
        max_disparity_train = []
        combinations = ["0|0", "0|1", "1|0", "1|1"]
        targets = ["0", "1"]

        sum_counters = {"0|0": 0, "0|1": 0, "1|0": 0, "1|1": 0}
        sum_targets = {"0": 0, "1": 0}

        for n_examples, node_metrics in metrics:
            losses.append(n_examples * node_metrics["train_loss"])
            losses_with_regularization.append(
                n_examples * node_metrics["train_loss_with_regularization"]
            )
            epsilon_list.append(node_metrics["epsilon"])
            accuracies.append(n_examples * node_metrics["train_accuracy"])
            lambda_list.append(node_metrics["Lambda"])
            disparity = node_metrics["Max Disparity Dataset"]
            client_id = node_metrics["cid"]
            disparity_client_after_local_epoch = node_metrics["Disparity Train"]
            max_disparity_train.append(disparity_client_after_local_epoch)
            disparity_client_before_local_epoch = node_metrics[
                "Max Disparity Train Before Local Epoch"
            ]

            # Load the lambda for the client
            # fed_dir = Path(fed_dir)
            # if os.path.exists(f"{fed_dir}/privacy_engine_{client_id}.pkl"):
            #     with open(f"{fed_dir}/DPL_lambda_{client_id}.pkl", "rb") as file:
            #         lambda_client = dill.load(file)

            DPL_lambda = node_metrics["DPL_lambda"]

            # load the statistics
            current_counter = node_metrics["counters"]
            for combination in combinations:
                sum_counters[combination] += current_counter[combination]
            for target in targets:
                sum_targets[target] += current_counter[target]

            # Create the dictionary we want to log. For some metrics we want to log
            # we have to check if they are present or not.
            to_be_logged = {
                f"Disparity Client {client_id} After Local train": disparity_client_after_local_epoch,
                f"Disparity Client {client_id} Before local train": disparity_client_before_local_epoch,
                "FL Round": server_round,
            }
            if disparity:
                to_be_logged[f"Disparity Dataset Client {client_id}"] = disparity
            if DPL_lambda:
                to_be_logged[f"Lambda Client {client_id}"] = DPL_lambda

            if wandb_run:
                wandb_run.log(
                    to_be_logged,
                )

        disparity_from_statistics = max(
            [
                sum_counters["0|0"] / sum_targets["0"]
                - sum_counters["0|1"] / sum_targets["1"],
                sum_counters["0|1"] / sum_targets["1"]
                - sum_counters["0|0"] / sum_targets["0"],
                sum_counters["1|0"] / sum_targets["0"]
                - sum_counters["1|1"] / sum_targets["1"],
                sum_counters["1|1"] / sum_targets["1"]
                - sum_counters["1|0"] / sum_targets["0"],
            ]
        )
        average_probabilities = {}
        for combination in combinations:
            average_probabilities[combination] = (
                sum_counters[combination] / sum_targets[combination[2]]
            )

        if wandb_run:
            wandb_run.log(
                {
                    "Training Disparity with statistics": disparity_from_statistics,
                    "FL Round": server_round,
                }
            )
        current_max_epsilon = max(current_max_epsilon, *epsilon_list)
        agg_metrics = {
            "Train Loss": sum(losses) / total_examples,
            "Train Accuracy": sum(accuracies) / total_examples,
            "Train Loss with Regularization": sum(losses_with_regularization)
            / total_examples,
            "Average Probabilities": average_probabilities,
            "Training Disparity with average": sum(max_disparity_train)
            / len(max_disparity_train),
            "Aggregated Lambda": sum(lambda_list) / len(lambda_list),
            "Train Epsilon": current_max_epsilon,
            "FL Round": server_round,
        }

        if wandb_run:
            wandb_run.log(
                agg_metrics,
            )

        return agg_metrics

    print(
        f"CLIENT SAMPLED: {args.sampled_clients}, {args.sampled_clients_validation}, {args.sampled_clients_test}"
    )
    strategy = FedAvg(
        fraction_fit=args.sampled_clients,
        fraction_evaluate=args.sampled_clients_validation,
        fraction_test=args.sampled_clients_test,
        min_fit_clients=args.sampled_clients,
        min_evaluate_clients=0,
        min_available_clients=args.sampled_clients,
        on_fit_config_fn=fit_config,
        # evaluate_fn=Utils.get_evaluate_fn(
        #     test_set=test_set,
        #     dataset_name=dataset_name,
        #     train_parameters=train_parameters,
        #     wandb_run=wandb_run,
        #     batch_size=args.batch_size,
        #     train_set=train_set,
        # ),  # centralised evaluation of global model
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=agg_metrics_train,
        evaluate_metrics_aggregation_fn=agg_metrics_evaluation,
        test_metrics_aggregation_fn=agg_metrics_test,
        current_max_epsilon=current_max_epsilon,
        fed_dir=fed_dir,
    )

    ray_num_cpus = 15
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

    num_training_nodes = int(args.pool_size * args.training_nodes)
    num_validation_nodes = int(args.pool_size * args.validation_nodes)
    num_test_nodes = int(args.pool_size * args.test_nodes)

    print(args.training_nodes, args.validation_nodes, args.test_nodes)
    print(num_training_nodes, num_validation_nodes, num_test_nodes)

    if num_training_nodes + num_validation_nodes + num_test_nodes != pool_size:
        raise Exception(
            "The sum of training, validation and test nodes must be equal to the pool size"
        )

    client_manager = SimpleClientManager(
        seed=args.seed,
        num_clients=pool_size,
        sort_clients=args.sort_clients,
        num_training_nodes=num_training_nodes,
        num_validation_nodes=num_validation_nodes,
        num_test_nodes=num_test_nodes,
        node_shuffle_seed=args.node_shuffle_seed,
        fed_dir=fed_dir,
    )
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
