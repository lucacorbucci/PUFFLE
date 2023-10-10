import argparse
import logging
import os
import random
import warnings
from pathlib import Path
from typing import Dict

import dill
import flwr as fl
import numpy as np
import torch
import wandb
from ClientManager.client_manager import SimpleClientManager
from DPL.Utils.dataset_utils import DatasetUtils
from DPL.Utils.model_utils import ModelUtils
from flwr.common.typing import Scalar
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
parser.add_argument("--wandb", type=bool, default=False)
parser.add_argument("--DPL", type=bool, default=False)
parser.add_argument("--DPL_lambda", type=float, default=0.0)
parser.add_argument("--private", type=bool, default=False)
parser.add_argument("--epsilon", type=float, default=None)
parser.add_argument("--noise_multiplier", type=float, default=0)
parser.add_argument("--clipping", type=float, default=1000000000)
parser.add_argument("--delta", type=float, default=None)
parser.add_argument("--lr", type=float, default="0.1")
parser.add_argument("--alpha", type=int, default=1000000)
parser.add_argument("--train_csv", type=str, default="")
parser.add_argument("--test_csv", type=str, default="")
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
parser.add_argument("--validation_ratio", type=float, default=0)
parser.add_argument("--optimizer", type=str, default=0)


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
            "probability_estimation": args.probability_estimation,
            "perfect_probability_estimation": args.perfect_probability_estimation,
            "alpha": args.alpha,
            "partition_ratio": args.partition_ratio,
            "alpha_target_lambda": args.alpha_target_lambda,
            "target": args.target,
            "weight_decay_lambda": args.weight_decay_lambda,
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
    print(f"SORT CLIENTS {args.sort_clients}")

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

    wandb_run = setup_wandb(args) if args.wandb else None
    # Download CIFAR-10 dataset
    train_set, test_set = DatasetUtils.download_dataset(
        dataset_name,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        debug=args.debug,
    )
    train_path = Utils.prepare_dataset_for_FL(
        dataset=train_set,
        dataset_name=dataset_name,
        base_path=args.base_path,
    )
    test_path = Utils.prepare_dataset_for_FL(
        dataset=test_set,
        dataset_name=dataset_name,
        base_path=args.base_path,
        partition="test"
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
        probability_estimation=args.probability_estimation,
        perfect_probability_estimation=args.perfect_probability_estimation,
        partition_ratio=args.partition_ratio,
        target=args.target,
        alpha=args.alpha_target_lambda,
        cross_silo=args.cross_silo,
        weight_decay_lambda=args.weight_decay_lambda,
        sweep=args.sweep,
    )

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.

    if (
        args.sweep
        and not args.validation_ratio
        or args.validation_ratio
        and not args.sweep
    ):
        raise Exception("When doing validation, sweep must be True and viceversa")

    # Partitioning the training dataset
    fed_dir = Utils.do_fl_partitioning(
        train_path,
        pool_size=pool_size,
        num_classes=2,
        val_ratio=args.validation_ratio,
        partition_type=args.partition_type,
        alpha=args.alpha,
        train_parameters=train_parameters,
    )

    if not args.validation_ratio:
        # Partitioning the test dataset
        fed_dir = Utils.do_fl_partitioning(
            test_path,
            pool_size=pool_size,
            num_classes=2,
            val_ratio=0,
            partition_type=args.partition_type,
            alpha=args.alpha,
            train_parameters=train_parameters,
            partition="test",
        )
        print(fed_dir)
    # fed_dir = "../data/celeba/celeba-10-batches-py/federated"

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
        disparities = max(
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
        if train_parameters.sweep:
            agg_metrics = {
                "Validation Loss": loss_evaluation,
                "Validation Accuracy": accuracy_evaluation,
                "Validation Disparity with average": max_disparity_average,
                "Validation Disparity with statistics": disparities,
                "FL Round": server_round,
            }
        else:
            agg_metrics = {
                "Test Loss": loss_evaluation,
                "Test Accuracy": accuracy_evaluation,
                "Test Disparity with average": max_disparity_average,
                "Test Disparity with statistics": disparities,
                "FL Round": server_round,
            }

        if wandb_run:
            wandb_run.log(agg_metrics)
        return agg_metrics

    def agg_metrics_train(metrics: list, server_round: int, current_max_epsilon: float, fed_dir) -> dict:
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
            fed_dir = Path(fed_dir)
            if os.path.exists(f"{fed_dir}/privacy_engine_{client_id}.pkl"):
                with open(f"{fed_dir}/DPL_lambda_{client_id}.pkl", "rb") as file:
                    lambda_client = dill.load(file)


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
            if lambda_client:
                to_be_logged[f"Lambda Client {client_id}"] = lambda_client

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
            average_probabilities[combination] = sum_counters[combination] / sum_targets[combination[2]]

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
            "Training Disparity with average": sum(max_disparity_train) / len(max_disparity_train),
            "Aggregated Lambda": sum(lambda_list) / len(lambda_list),
            "Train Epsilon": current_max_epsilon,
            "FL Round": server_round,
        }

        if wandb_run:
            wandb_run.log(
                agg_metrics,
            )

        return agg_metrics

            # losses_node = [
            #     n_examples * metric for metric in node_metrics["train_losses"]
            # ]
            # all_losses.append(losses_node)

        # all_losses = np.array(all_losses)
        # sum_losses = np.sum(all_losses, axis=0)

        # if wandb_run:
        #     for index, loss in enumerate(sum_losses):
        #         wandb_run.log(
        #             {
        #                 "Loss Epochs": loss,
        #                 "Epoch": (server_round - 1) * len(sum_losses) + index,
        #             }
        #         )
        
        # Collect all the metrics that can be directly extracted from the metric
        # dictionary 
        # losses = [n_examples * metric["train_loss"] for n_examples, metric in metrics]
        # losses_with_regularization = [
        #     n_examples * metric["train_loss_with_regularization"]
        #     for n_examples, metric in metrics
        # ]
        # epsilon_list = [metric["epsilon"] for _, metric in metrics]
        # accuracies = [
        #     n_examples * metric["train_accuracy"] for n_examples, metric in metrics
        # ]
        # lambda_list = [metric["Lambda"] for _, metric in metrics]

        # max_disparity_train = [
        #     metric["Disparity Train"] for n_examples, metric in metrics
        # ]

        # # Log information regarding the single nodes
        # for n_examples, metric in metrics:
            

        # fed_dir = "../data/celeba/celeba-10-batches-py/federated"
        # Now we can log the values of the lambda for each client
        # fed_dir = Path(fed_dir)
        # cid_list = [i for i in range(args.pool_size)]
        # for cid in cid_list:
            

        # total_examples = sum([n_examples for n_examples, _ in metrics])

        # possible_targets = []
        # possible_sensitive_attributes = []

        # for _, metric in metrics:
        #     possible_targets += metric["targets"]
        #     possible_sensitive_attributes += metric["sensitive_attributes"]

        # possible_targets = list(set(possible_targets))
        # possible_sensitive_attributes = list(set(possible_sensitive_attributes))

        # sum_probabilities = {}
        # total_counter = {}

        # for _, metric in metrics:
        #     for possible_target in possible_targets:
        #         for possible_sensitive_attribute in possible_sensitive_attributes:
        #             current_proba = f"{possible_target}|{possible_sensitive_attribute}"

        #             if current_proba in metric["probabilities"]:
        #                 if current_proba not in sum_probabilities:
        #                     sum_probabilities[current_proba] = 0
        #                 sum_probabilities[current_proba] += metric["probabilities"][
        #                     current_proba
        #                 ]

        #     for possible_sensitive_attribute in possible_sensitive_attributes:
        #         attr = f"{possible_sensitive_attribute}"
        #         if attr in metric["probabilities"]:
        #             if attr not in total_counter:
        #                 total_counter[attr] = 0
        #             total_counter[f"{possible_sensitive_attribute}"] += metric[
        #                 "probabilities"
        #             ][attr]


        # We consider the statistics about the predictions shared by the clients
        # we want to count the predictions for each possible combination 
        # and we want to count the number of targets for each possible target

    strategy = FedAvg(
        fraction_fit=args.sampled_clients,
        fraction_evaluate=args.sampled_clients,
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
            train_set=train_set,
        ),  # centralised evaluation of global model
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=agg_metrics_train,
        evaluate_metrics_aggregation_fn=agg_metrics_evaluation,
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

    client_manager = SimpleClientManager(
        seed=args.seed, num_clients=pool_size, sort_clients=args.sort_clients
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
