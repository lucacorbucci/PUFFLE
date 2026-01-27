import argparse
import os
import shutil
import signal
import sys
import time
from typing import Any

import wandb
from datasets import load_dataset
from flwr.client import ClientApp
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from flwr_datasets.visualization import plot_label_distributions

from FlowerFLTemplate.Aggregations.aggregations import Aggregation
from FlowerFLTemplate.ClientManager.client_manager import SimpleClientManager
from FlowerFLTemplate.Datasets.dataset_utils import (
    get_data_info,
    prepare_data_for_cross_device,
    prepare_data_for_cross_silo,
)
from FlowerFLTemplate.Models.utils import get_model
from FlowerFLTemplate.Server.server import Server
from FlowerFLTemplate.Strategy.fed_avg import FedAvg
from FlowerFLTemplate.Utils.preferences import Preferences
from FlowerFLTemplate.Utils.utils import get_params, seed_everything


def signal_handler(sig: int, frame: Any) -> None:
    """
    Handles interrupt signals to gracefully terminate the experiment.

    Finishes the Weights & Biases run if active and exits the program cleanly.

    Args:
        sig (int): The signal number received.
        frame (Any): The current stack frame.

    Returns:
        None

    """
    print("Gracefully stopping your experiment! Keep calm!")
    print("Gracefully stopping your experiment! Keep calm!")
    if wandb_run:
        wandb_run.finish()
    sys.exit(0)


def client_fn(context: Context) -> Any:
    """
    Generates a Flower client instance with its assigned data partition.

    Loads the partition based on the global partitioner and prepares data for the specified FL setting (cross-device or cross-silo).

    Args:
        context (Context): The Flower context with node configuration including partition ID.

    Returns:
        Any: A configured Flower client instance.

    Raises:
        KeyError: If "partition-id" is not found in node_config.

    """
    partition_id = int(context.node_config["partition-id"])
    partition = partitioner.load_partition(partition_id) if partitioner else None

    if preferences is None:
        msg = "Preferences not initialized"
        raise ValueError(msg)

    if preferences.cross_device:
        return prepare_data_for_cross_device(
            context, partition, preferences, partition_id
        )

    return prepare_data_for_cross_silo(context, partition, preferences, partition_id)


def server_fn(context: Context) -> ServerAppComponents:
    """
    Constructs ServerAppComponents for running the Flower server simulation.

    Initializes the global model, defines the FedAvg strategy with aggregation functions, and sets up the server with client manager and preferences.

    Args:
        context (Context): The Flower context.

    Returns:
        ServerAppComponents: Components including the server instance and configuration.

    """
    if preferences is None:
        msg = "Preferences not initialized"
        raise ValueError(msg)

    # Instantiate the model
    # Determine model name from dataset if not explicitly set
    model_name = preferences.model
    if model_name is None:
        if preferences.dataset_name == "dutch":
            model_name = "LinearClassificationNet"
        elif preferences.dataset_name == "abalone":
            model_name = "AbaloneNet"
        elif preferences.dataset_name == "mnist":
            model_name = "SimpleMNISTModel"
        elif preferences.dataset_name == "celeba":
            model_name = "CelebaNet"
        else:
            msg = f"Unknown dataset for model selection: {preferences.dataset_name}"
            raise ValueError(msg)

    model = get_model(
        model_name=model_name,
        num_classes=preferences.num_classes,
        in_channels=preferences.in_channels,
        pixel=preferences.pixel,
    )

    ndarrays = get_params(model)
    # Convert model parameters to flwr.common.Parameters
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = FedAvg(
        fraction_fit=preferences.sampled_training_nodes_per_round
        if preferences.sampled_training_nodes_per_round
        else 0.1,
        fraction_evaluate=preferences.sampled_validation_nodes_per_round
        if preferences.sampled_validation_nodes_per_round
        and preferences.sampled_validation_nodes_per_round > 0
        else (
            preferences.sampled_test_nodes_per_round
            if preferences.sampled_test_nodes_per_round
            else 0.0
        ),
        initial_parameters=global_model_init,  # initialised global model
        fit_metrics_aggregation_fn=Aggregation.agg_metrics_train,
        evaluate_metrics_aggregation_fn=Aggregation.agg_metrics_evaluation,
        test_metrics_aggregation_fn=Aggregation.agg_metrics_test,
        preferences=preferences,
        wandb_run=wandb_run,
    )

    config = ServerConfig(num_rounds=preferences.num_rounds or 1)

    if client_manager is None:
        # Should not happen as initialized in main before server_app
        # But needed for type checker
        msg = "Client Manager not initialized"
        raise ValueError(msg)

    server = Server(
        client_manager=client_manager, strategy=strategy, preferences=preferences
    )

    # Wrap everything into a `ServerAppComponents` object
    return ServerAppComponents(server=server, config=config)  # type: ignore


def get_partitioner(preferences: Preferences) -> Any:
    """
    Returns a partitioner based on the specified type in preferences.
    Supports "iid" and "non_iid" (Dirichlet) partitioning.

    Args:
        preferences (Preferences): User preferences containing partitioner settings.

    Returns:
        Any: An instance of the selected partitioner.

    Raises:
        ValueError: If an unsupported partitioner type is specified.

    """
    partitioner_type = preferences.partitioner_type

    match partitioner_type:
        case "iid":
            return IidPartitioner(
                num_partitions=preferences.num_clients or 10
            )  # default if None
        case "non_iid":
            if preferences.partitioner_by is None:
                msg = "partitioner_by must be set for non_iid partitioning"
                raise ValueError(msg)
            return DirichletPartitioner(
                num_partitions=preferences.num_clients or 10,
                alpha=preferences.partitioner_alpha,
                partition_by=preferences.partitioner_by,
            )
        case _:
            error = f"Unsupported partitioner type: {partitioner_type}"
            raise ValueError(error)


def prepare_data(preferences: Preferences) -> Any:
    """
    Loads and prepares the dataset based on the specified name in preferences.

    Supports datasets: "dutch", "mnist", "abalone", "income". Sets up scaler/encoder if applicable, creates partitioner, and optionally plots label distributions.

    Args:
        preferences (Preferences): User preferences containing dataset settings.

    Returns:
        Any: The partitioner instance for data partitioning, or None for "income" dataset.

    Raises:
        ValueError: If an unsupported dataset is specified or no training data is found.

    """
    if preferences.dataset_name == "dutch":
        data_info = get_data_info(preferences)
        preferences.scaler = data_info.get("scaler", None)
        dataset_dict = load_dataset("csv", data_files=preferences.dataset_path)
    elif preferences.dataset_name == "mnist":
        data_info = get_data_info(preferences)
        dataset_dict = load_dataset(
            data_info["data_type"], data_dir=preferences.dataset_path
        )
    elif preferences.dataset_name == "abalone":
        data_info = get_data_info(preferences)
        preferences.scaler = data_info.get("scaler", None)
        dataset_dict = load_dataset("csv", data_files=preferences.dataset_path)
    elif preferences.dataset_name == "income":
        data_info = get_data_info(preferences)
        preferences.scaler = data_info.get("scaler", None)
        preferences.encoder = data_info.get("encoder", None)
        partitioner = None
        return partitioner
    elif preferences.dataset_name == "celeba":
        data_info = get_data_info(preferences)
        dataset_dict = load_dataset("csv", data_files=preferences.dataset_path)
    # elif preferences.dataset_name == "speech_fairness":

    else:
        error = f"Unsupported dataset: {preferences.dataset_name}"
        raise ValueError(error)

    data = dataset_dict.get("train", None)  # type: ignore
    if data:
        partitioner = get_partitioner(preferences)
        partitioner.dataset = data
    else:
        error = "No training data found in the dataset"
        raise ValueError(error)

    if preferences.partitioner_by:
        plot, _, _ = plot_label_distributions(
            partitioner=partitioner,
            label_name=preferences.partitioner_by,
            plot_type="bar",
            size_unit="absolute",
            partition_id_axis="x",
            legend=True,
            verbose_labels=True,
            max_num_partitions=preferences.num_clients,
            title="Per Partition Labels Distribution",
        )
        plot.savefig(
            f"label_distribution_{preferences.partitioner_by}_{preferences.partitioner_type}.png",
            bbox_inches="tight",
        )

    return partitioner


def setup_wandb(project_name: str, run_name: str | None) -> Any:
    """
    Initializes a Weights & Biases (wandb) run for experiment tracking.

    Args:
        project_name (str): The name of the wandb project.
        run_name (str | None): The name of the specific run; uses default if None.

    Returns:
        Any: The initialized wandb run object.

    Raises:
        Exception: If wandb initialization fails due to configuration or network issues.

    """
    return (
        wandb.init(project=project_name, name=run_name)
        if run_name
        else wandb.init(project=project_name)
    )


parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument("--num_clients", type=int, default=None, required=True)
parser.add_argument("--num_rounds", type=int, default=None, required=True)
parser.add_argument("--num_epochs", type=int, default=None)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--FL_setting", type=str, default=None, required=True)
parser.add_argument("--dataset_name", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--node_shuffle_seed", type=int, default=None)
parser.add_argument("--partitioner_type", type=str, default="iid")
parser.add_argument("--partitioner_alpha", type=float, default=None)
parser.add_argument("--partitioner_by", type=str, default=None)
parser.add_argument("--num_test_nodes", type=int, default=None)
parser.add_argument("--num_validation_nodes", type=int, default=None)
parser.add_argument("--num_train_nodes", type=int, default=None)
parser.add_argument("--sampled_validation_nodes_per_round", type=float, default=None)
parser.add_argument("--sampled_train_nodes_per_round", type=float, default=None)
parser.add_argument("--sampled_test_nodes_per_round", type=float, default=None)
parser.add_argument("--fed_dir", type=str, default=None, required=True)
parser.add_argument("--dataset_path", type=str, default=None)
parser.add_argument("--sweep", type=bool, default=False)
parser.add_argument("--wandb", type=bool, default=True)
parser.add_argument("--project_name", type=str, default="FlowerFLTemplate")
parser.add_argument("--run_name", type=str, default=None)

parser.add_argument("--task", type=str, default="classification")


parser.add_argument("--image_path", type=str, default=None)

# Unfairness reduction parameters
parser.add_argument(
    "--unfairness_reduction", type=lambda x: str(x).lower() == "true", default=False
)
parser.add_argument("--regularization_lambda", type=float, default=0.0)
parser.add_argument("--fairness_metric", type=str, default="disparity")
parser.add_argument("--regularization_mode", type=str, default="fixed")
parser.add_argument("--target", type=float, default=None)
parser.add_argument("--alpha", type=float, default=None)
parser.add_argument("--weight_decay_alpha", type=float, default=None)

# Privacy parameters
parser.add_argument("--epsilon", type=float, default=None)
parser.add_argument("--noise_multiplier", type=float, default=0.0)
parser.add_argument("--max_grad_norm", type=float, default=1000000.0)

# Initialize global variables for client_fn/server_fn access
preferences: Preferences | None = None
partitioner: Any = None
client_manager: SimpleClientManager | None = None
wandb_run: Any = None


def main():
    signal.signal(signal.SIGINT, signal_handler)
    # remove files in tmp/ray
    args = parser.parse_args()

    if args.node_shuffle_seed is None:
        node_shuffle_seed = int(str(time.time()).split(".")[1]) * args.seed
        args.node_shuffle_seed = node_shuffle_seed
    seed_everything(args.seed)

    num_clients = args.num_clients
    num_rounds = args.num_rounds

    cross_device = args.FL_setting == "cross_device"

    # Global preferences object
    global preferences  # noqa: PLW0603
    preferences = Preferences(
        num_clients=num_clients,
        num_rounds=num_rounds,
        cross_device=cross_device,
        num_test_nodes=args.num_test_nodes,
        num_validation_nodes=args.num_validation_nodes,
        num_train_nodes=args.num_train_nodes,
        num_epochs=args.num_epochs,
        sampled_validation_nodes_per_round=args.sampled_validation_nodes_per_round,
        sampled_training_nodes_per_round=args.sampled_train_nodes_per_round,
        sampled_test_nodes_per_round=args.sampled_test_nodes_per_round,
        seed=args.seed,
        node_shuffle_seed=args.node_shuffle_seed,
        fed_dir=args.fed_dir,
        fl_setting=args.FL_setting,
        sweep=args.sweep,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        partitioner_type=args.partitioner_type,
        partitioner_alpha=args.partitioner_alpha,
        partitioner_by=args.partitioner_by,
        batch_size=args.batch_size,
        lr=args.lr,
        optimizer=args.optimizer,
        momentum=args.momentum,
        task=args.task,
        image_path=args.image_path,
        weight_decay=args.weight_decay,
        weight_decay_alpha=args.weight_decay_alpha,
        alpha=args.alpha,
        regularization_lambda=args.regularization_lambda,
        unfairness_reduction=args.unfairness_reduction,
        fairness_metric=args.fairness_metric,
        regularization_mode=args.regularization_mode,
        target=args.target,
        epsilon=args.epsilon,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
    )

    # Needs to be global for client_fn/server_fn to access?
    # client_fn and server_fn use `preferences` from outer scope.
    # To avoid global variable mess, we should pass preferences to them or use partials,
    # but Flwr API expects distinct signatures.
    # For now, using global scope or `global preferences` inside main is not enough if they are defined outside.
    # The functions server_fn and client_fn are defined at module level and use `preferences`.
    # Make sure `preferences` is available to them.
    # We can inject it or use a closure?
    # Simple fix: `global preferences` in main() writes to module-level variable.
    # But `preferences` is not defined at module level yet.
    # I should define `preferences = None` at top level.

    # remove the files in the path args.fed_dir
    if os.path.exists(args.fed_dir):
        for item in os.listdir(args.fed_dir):
            item_path = os.path.join(args.fed_dir, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)  # remove file or symlink
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # remove directory
    else:
        # Create it if it doesn't exist (though load_partitioned_dataset might do it, better safe)
        os.makedirs(args.fed_dir, exist_ok=True)

    global wandb_run  # noqa: PLW0603
    wandb_run = (
        setup_wandb(
            project_name=args.project_name,
            run_name=args.run_name,
        )
        if args.wandb
        else None
    )

    # Create your ServerApp
    global client_manager  # noqa: PLW0603
    client_manager = SimpleClientManager(preferences=preferences)

    global partitioner  # noqa: PLW0603
    partitioner = prepare_data(preferences=preferences)

    # Create your ServerApp
    server_app = ServerApp(server_fn=server_fn)

    # Concstruct the ClientApp passing the client generation function
    client_app = ClientApp(client_fn=client_fn)

    run_simulation(
        server_app=server_app, client_app=client_app, num_supernodes=num_clients
    )

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
