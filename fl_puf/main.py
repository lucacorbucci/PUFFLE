import argparse
import logging
import os
import random
import warnings
from typing import Dict

import flwr as fl
import numpy as np
import torch
from ClientManager.client_manager import SimpleClientManager
from Server.server import Server
from Strategy.fed_avg import FedAvg
from Utils.train_parameters import TrainParameters
from flwr.common.typing import Scalar
from opacus import PrivacyEngine
from torch import nn

from DPL.Utils.dataset_utils import DatasetUtils
from DPL.Utils.model_utils import ModelUtils
from fl_puf.Client.client import FlowerClient
from fl_puf.Utils.tabular_data_loader import prepare_tabular_data
from fl_puf.Utils.utils import Utils

warnings.filterwarnings("ignore")

# ----------------------------- Parsing Options ----------------------------------------
parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

# ----------------------
# Training Settings
parser.add_argument(
    "--num_client_cpus", type=float, default=1
)  # Percentage of CPUs used by each client
parser.add_argument(
    "--num_client_gpus", type=float, default=1
)  # Percentage of GPUs used by each client
parser.add_argument(
    "--num_rounds", type=int, default=5
)  # Number of rounds of federated learning
parser.add_argument("--dataset", type=str, default=None)  # Dataset we want to use
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
    "--DPL", type=bool, default=False
)  # If we want to use DPL Regularization
parser.add_argument("--private", type=bool, default=False)  # If we want to use DP-SGD
parser.add_argument("--epsilon", type=float, default=None)  # Target Epsilon for DP-SGD
parser.add_argument(
    "--noise_multiplier", type=float, default=0
)  # Noise multiplier for DP-SGD
parser.add_argument(
    "--clipping", type=float, default=1000000000
)  # Clipping value for DP-SGD
parser.add_argument("--delta", type=float, default=None)

# ----------------------
# Dataset/Distribution Settings
parser.add_argument("--train_csv", type=str, default="")
parser.add_argument("--base_path", type=str, default="")
parser.add_argument("--sort_clients", type=bool, default=True)
parser.add_argument("--no_sort_clients", dest="sort_clients", action="store_false")
parser.add_argument(
    "--node_shuffle_seed", type=int, default=30
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


# ----------------------
# Parameters for the Lambda
parser.add_argument(
    "--starting_lambda_mode", type=str, default=None
)  # how to initialize the Lambda
parser.add_argument(
    "--starting_lambda_value", type=float, default=None
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
)  # percentage of samples removed from group_to_reduce on the unfair nodes when opposite_direction is true
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
    "--sampling_frequency", type=int, default=None
)  # The number of times that each node will be sampled


# --------------------------------------------------------------------------------------

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

    if args.tabular_data:
        fed_dir, _ = prepare_tabular_data(
            dataset_path=args.dataset_path,
            dataset_name=dataset_name,
            groups_balance_factor=args.groups_balance_factor,
            priv_balance_factor=args.priv_balance_factor,
            do_iid_split=True,
            approach=args.approach,
            num_nodes=args.pool_size,
            ratio_unfair_nodes=args.ratio_unfair_nodes,
            opposite_direction=args.opposite_direction,
            ratio_unfairness=tuple(args.ratio_unfairness)
            if args.ratio_unfairness
            else None,
            group_to_reduce=tuple(args.group_to_reduce)
            if args.group_to_reduce
            else None,
            group_to_increment=tuple(args.group_to_increment)
            if args.group_to_increment
            else None,
            number_of_samples_per_node=args.number_of_samples_per_node,
            opposite_group_to_reduce=tuple(args.opposite_group_to_reduce)
            if args.opposite_group_to_reduce
            else None,
            opposite_group_to_increment=tuple(args.opposite_group_to_increment)
            if args.opposite_group_to_increment
            else None,
            opposite_ratio_unfairness=tuple(args.opposite_ratio_unfairness)
            if args.opposite_ratio_unfairness
            else None,
        )

    else:
        train_set, test_set = DatasetUtils.download_dataset(
            dataset_name,
            train_csv=args.train_csv,
            debug=args.debug,
            base_path=args.dataset_path,
        )
        train_path = Utils.prepare_dataset_for_FL(
            dataset=train_set,
            dataset_name=dataset_name,
            base_path=args.base_path,
        )

    DPL_value = None
    if args.starting_lambda_mode == "fixed" and args.starting_lambda_value is None:
        raise Exception("Starting lambda value must be specified when using fixed mode")
    if args.starting_lambda_mode == "no_tuning":
        DPL_value = 0
        args.starting_lambda_value = 0
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
        percentage_unbalanced_nodes=args.percentage_unbalanced_nodes,
        target=args.target,
        alpha=args.alpha_target_lambda,
        cross_silo=args.cross_silo,
        weight_decay_lambda=args.weight_decay_lambda,
        sweep=args.sweep,
        starting_lambda_mode=args.starting_lambda_mode,
        starting_lambda_value=args.starting_lambda_value,
        momentum=args.momentum,
        update_lambda=args.update_lambda,
        unbalanced_ratio=args.unbalanced_ratio,
        tabular_data=args.tabular_data,
        sampling_frequency=args.sampling_frequency,
    )

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.

    if not args.tabular_data:
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

    # if args.epsilon:
    #     # We need to understand the noise that we need to add based
    #     # on the epsilon that we want to guarantee
    #     max_noise = 0
    #     for i in range(args.pool_size):
    #         model_noise = ModelUtils.get_model(
    #             dataset_name, device=train_parameters.device
    #         )
    #         # get the training dataset of one of the clients
    #         train_loader_client_0 = Utils.get_dataloader(
    #             fed_dir,
    #             str(i),
    #             batch_size=train_parameters.batch_size,
    #             workers=0,
    #             dataset=dataset_name,
    #             partition="train",
    #         )
    #         privacy_engine = PrivacyEngine(accountant="rdp")
    #         optimizer_noise = Utils.get_optimizer(
    #             model_noise, train_parameters, args.lr
    #         )
    #         (
    #             _,
    #             private_optimizer,
    #             _,
    #         ) = privacy_engine.make_private_with_epsilon(
    #             module=model_noise,
    #             optimizer=optimizer_noise,
    #             data_loader=train_loader_client_0,
    #             epochs=args.sampling_frequency * args.epochs,
    #             target_epsilon=train_parameters.epsilon,
    #             target_delta=args.delta,
    #             max_grad_norm=args.clipping,
    #         )
    #         max_noise = max(max_noise, private_optimizer.noise_multiplier)
    #         print(
    #             f"Node {i} - {args.sampling_frequency * args.epochs} -- {private_optimizer.noise_multiplier}"
    #         )

    #     train_parameters.noise_multiplier = max_noise
    #     train_parameters.epsilon = None
    #     print(
    #         f">>>>> FINALE {args.sampling_frequency * args.epochs} -- {train_parameters.noise_multiplier}"
    #     )
    # else:
    #     train_parameters.noise_multiplier = args.noise_multiplier
    #     train_parameters.epsilon = None

    wandb_run = Utils.setup_wandb(args, train_parameters) if args.wandb else None

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
        f1_test = (
            sum([n_examples * metric["f1_score"] for n_examples, metric in metrics])
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
        # weighted average of the disparity of the different nodes
        max_disparity_weighted_average = (
            sum(
                [
                    n_examples
                    * metric[
                        "max_disparity_test"
                        if not train_parameters.sweep
                        else "max_disparity_validation"
                    ]
                    for n_examples, metric in metrics
                ]
            )
            / total_examples
        )

        # Log data from the different test clients:
        for _, metric in metrics:
            node_name = metric["cid"]
            disparity = metric[
                "max_disparity_test"
                if not train_parameters.sweep
                else "max_disparity_validation"
            ]
            accuracy = metric[
                "test_accuracy" if not train_parameters.sweep else "validation_accuracy"
            ]
            disparity_dataset = metric["max_disparity_dataset"]
            agg_metrics = {
                f"Test Node {node_name} - Acc.": accuracy,
                f"Test Node {node_name} - Disp.": disparity,
                f"Test Node {node_name} - Disp. Dataset": disparity_dataset,
                "FL Round": server_round,
            }
            if wandb_run:
                wandb_run.log(agg_metrics)

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

        sum_counters_no_noise = {"0|0": 0, "0|1": 0, "1|0": 0, "1|1": 0}
        sum_targets_no_noise = {"0": 0, "1": 0}

        for _, metric in metrics:
            metric = metric["counters_no_noise"]
            for combination in combinations:
                sum_counters_no_noise[combination] += metric[combination]
            for target in targets:
                sum_targets_no_noise[target] += metric[target]
        max_disparity_statistics_no_noise = max(
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

        agg_metrics = {
            "Test Loss": loss_test,
            "Test Accuracy": accuracy_test,
            "Test Disparity with average": max_disparity_average,
            "Test Disparity with weighted average": max_disparity_weighted_average,
            "Test Disparity with statistics": max_disparity_statistics,
            "Test Disparity with statistics no noise": max_disparity_statistics_no_noise,
            "FL Round": server_round,
            "Test Counter 0|0": sum_counters["0|0"],
            "Test Counter 0|1": sum_counters["0|1"],
            "Test Counter 1|0": sum_counters["1|0"],
            "Test Counter 1|1": sum_counters["1|1"],
            "Test Target 0": sum_targets["0"],
            "Test Target 1": sum_targets["1"],
            "Test F1": f1_test,
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
        f1_validation = (
            sum([n_examples * metric["f1_score"] for n_examples, metric in metrics])
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
        # weighted average of the disparity of the different nodes
        max_disparity_weighted_average = (
            sum(
                [
                    n_examples
                    * metric[
                        "max_disparity_test"
                        if not train_parameters.sweep
                        else "max_disparity_validation"
                    ]
                    for n_examples, metric in metrics
                ]
            )
            / total_examples
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

        sum_counters_no_noise = {"0|0": 0, "0|1": 0, "1|0": 0, "1|1": 0}
        sum_targets_no_noise = {"0": 0, "1": 0}

        for _, metric in metrics:
            metric = metric["counters_no_noise"]
            for combination in combinations:
                sum_counters_no_noise[combination] += metric[combination]
            for target in targets:
                sum_targets_no_noise[target] += metric[target]
        max_disparity_statistics_no_noise = max(
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
            "Validation Disparity with weighted average": max_disparity_weighted_average,
            "Validation Disparity with statistics": max_disparity_statistics,
            "Validation Disparity with statistics no noise": max_disparity_statistics_no_noise,
            "Custom_metric": custom_metric,
            "FL Round": server_round,
            "Validation Counter 0|0": sum_counters["0|0"],
            "Validation Counter 0|1": sum_counters["0|1"],
            "Validation Counter 1|0": sum_counters["1|0"],
            "Validation Counter 1|0": sum_counters["1|0"],
            "Validation Counter 1|1": sum_counters["1|1"],
            "Validation Target 0": sum_targets["0"],
            "Validation Target 1": sum_targets["1"],
            "Validation F1": f1_validation,
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

        average_probabilities = {}
        for combination in combinations:
            average_probabilities[combination] = (
                sum_counters[combination] / sum_targets[combination[2]]
            )

        # weighted average of the disparity of the different nodes
        max_disparity_weighted_average = (
            sum(
                [
                    n_examples * metric["Disparity Train"]
                    for n_examples, metric in metrics
                ]
            )
            / total_examples
        )

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

        sum_counters_no_noise = {"0|0": 0, "0|1": 0, "1|0": 0, "1|1": 0}
        sum_targets_no_noise = {"0": 0, "1": 0}

        for _, metric in metrics:
            metric = metric["counters_no_noise"]
            for combination in combinations:
                sum_counters_no_noise[combination] += metric[combination]
            for target in targets:
                sum_targets_no_noise[target] += metric[target]
        max_disparity_statistics_no_noise = max(
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

        if wandb_run:
            wandb_run.log(
                {
                    "Training Disparity with statistics": max_disparity_statistics,
                    "Training Disparity with statistics no noise": max_disparity_statistics_no_noise,
                    "FL Round": server_round,
                    "Training Counter 0|0": sum_counters["0|0"],
                    "Training Counter 0|1": sum_counters["0|1"],
                    "Training Counter 1|0": sum_counters["1|0"],
                    "Training Counter 1|1": sum_counters["1|1"],
                    "Training Target 0": sum_targets["0"],
                    "Training Target 1": sum_targets["1"],
                }
            )

        print(
            f"LOSS WITH REGULARIZATION {sum(losses_with_regularization) / total_examples}"
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
            "Training Disparity with weighted average": max_disparity_weighted_average,
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
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=initial_parameters,
        fit_metrics_aggregation_fn=agg_metrics_train,
        evaluate_metrics_aggregation_fn=agg_metrics_evaluation,
        test_metrics_aggregation_fn=agg_metrics_test,
        current_max_epsilon=current_max_epsilon,
        fed_dir=fed_dir,
    )

    ray_num_cpus = 20
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
