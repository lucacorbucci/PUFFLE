import argparse
import logging
import os
import shutil
import signal
import sys
import time
from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
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
from FlowerFLTemplate.ClientManager.fairness_client_manager import FairnessClientManager
from FlowerFLTemplate.Datasets.dataset_utils import (
    get_data_info,
    get_model_info_from_dataset,
    prepare_data_for_cross_device,
    prepare_data_for_cross_silo,
)
from FlowerFLTemplate.Datasets.dutch import prepare_dutch_for_fairness
from FlowerFLTemplate.Datasets.Partitioner.fairness_partitioner import (
    FairnessPartitioner,
)
from FlowerFLTemplate.Models.utils import get_model
from FlowerFLTemplate.Server.server import Server
from FlowerFLTemplate.Strategy.fed_avg import FedAvg
from FlowerFLTemplate.Utils.preferences import Preferences
from FlowerFLTemplate.Utils.utils import get_params, seed_everything

# hide ray logs
os.environ["RAY_LOG_LEVEL"] = "ERROR"

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
    if wandb_run:
        wandb_run.finish()
    os._exit(0)


def client_fn(context: Context) -> Any:
    """
    Generates a Flower client instance with its assigned data partition.

    Uses lazy loading: data is not loaded until the first fit()/evaluate() call.
    This allows get_properties() to respond instantly during registration.

    Args:
        context (Context): The Flower context with node configuration including partition ID.

    Returns:
        Any: A configured Flower client instance.

    Raises:
        KeyError: If "partition-id" is not found in node_config.

    """
    partition_id = int(context.node_config["partition-id"])

    if preferences is None:
        msg = "Preferences not initialized"
        raise ValueError(msg)

    if preferences.cross_device:
        # Use lazy loading: pass partitioner instead of loading partition now
        return prepare_data_for_cross_device(
            context,
            partition=None,  # Don't load partition now
            preferences=preferences,
            partition_id=partition_id,
            partitioner=partitioner,  # Pass partitioner for lazy loading
        )

    # Cross-silo still loads data eagerly (more complex train/val splits)
    partition = partitioner.load_partition(partition_id) if partitioner else None
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
        msg = "Model name is not specified."
        raise ValueError(msg)

    model = get_model(
        model_name=model_name,
        num_classes=preferences.num_classes,
        in_channels=preferences.in_channels,
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
        target=preferences.target,
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


def plot_fairness_distributions(
    title: str, counter_groups: list, all_combinations: list, filename: str
):
    """Plot stacked bar chart showing distribution of (target, sensitive) groups per client."""
    plt.figure(figsize=(20, 8))
    previous_sum = []

    # Iterate through each group combination (e.g. (0,0), (0,1), etc.)
    for combination in all_combinations:
        # Extract count for this combination for each client
        counter = [counter.get(combination, 0) for counter in counter_groups]

        if previous_sum:
            plt.bar(
                range(len(counter)),
                counter,
                bottom=previous_sum,
                label=str(combination),
            )
        else:
            plt.bar(range(len(counter)), counter, label=str(combination))
            previous_sum = [0] * len(counter)

        # Update previous_sum for stacking
        previous_sum = [sum(x) for x in zip(previous_sum, counter, strict=False)]

    plt.xlabel("Client")
    plt.ylabel("Amount of samples")
    plt.title(title)
    plt.legend()
    plt.rcParams.update({"font.size": 15})
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


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
        case "fairness":
            if (
                preferences.sensitive_attribute is None
                or preferences.target_attribute is None
            ):
                msg = "sensitive_attribute and target_attribute are required for fairness partitioning"
                raise ValueError(msg)
            return FairnessPartitioner(
                num_partitions=preferences.num_clients or 10,
                sensitive_attribute=preferences.sensitive_attribute,
                target_attribute=preferences.target_attribute,
                ratio_unfair_clients=preferences.ratio_unfair_clients
                if preferences.ratio_unfair_clients is not None
                else 0.5,
                group_to_reduce=preferences.group_to_reduce,
                ratio_unfairness=preferences.ratio_unfairness
                if preferences.ratio_unfairness
                else (0.8, 0.9),
                group_to_increment=preferences.group_to_increment,
                seed=preferences.seed,
                samples_per_client=preferences.samples_per_client,
                distribution_mode=preferences.distribution_mode,
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
    data = None
    if preferences.dataset_name == "dutch":
        data_info = get_data_info(preferences)
        preferences.scaler = data_info.get("scaler", None)
        dataset_dict = load_dataset("csv", data_files=preferences.dataset_path)
        if preferences.partitioner_type == "fairness":
            data = prepare_dutch_for_fairness(preferences, dataset_dict, 0)
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
    else:
        error = f"Unsupported dataset: {preferences.dataset_name}"
        raise ValueError(error)

    if not data:
        data = dataset_dict.get("train", None)  # type: ignore

    if data:
        partitioner = get_partitioner(preferences)
        partitioner.dataset = data
    else:
        error = "No training data found in the dataset"
        raise ValueError(error)

    if preferences.partitioner_by or preferences.partitioner_type == "fairness":
        # Determine label name with fallback
        label_name = (
            preferences.partitioner_by
            if preferences.partitioner_by
            else preferences.target_attribute
        )
        if label_name is None:
            label_name = "label"  # Fallback if both are None

        plot, _, _ = plot_label_distributions(
            partitioner=partitioner,
            label_name=label_name,
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

        plot, _, _ = plot_label_distributions(
            partitioner=partitioner,
            label_name="sex",
            plot_type="bar",
            size_unit="absolute",
            partition_id_axis="x",
            legend=True,
            verbose_labels=True,
            max_num_partitions=preferences.num_clients,
            title="Per Partition Labels Distribution",
        )
        plot.savefig(
            f"label_distribution_sex_{preferences.partitioner_type}.png",
            bbox_inches="tight",
        )

        # Plot fairness-specific distributions if using fairness partitioner
        if preferences.partitioner_type == "fairness":
            # Collect counts per client
            counter_groups = []

            # Get the actual dataset to determine unique groups
            sample_partition = partitioner.load_partition(0).to_pandas()

            # Identify all possible groups from the dataset
            unique_groups = sorted(
                set(
                    zip(
                        sample_partition[preferences.target_attribute],
                        sample_partition[preferences.sensitive_attribute],
                        strict=False,
                    )
                )
            )
            print(f"All unique groups found: {unique_groups}")

            if preferences.num_clients is None:
                msg = "num_clients must be set"
                raise ValueError(msg)
            for i in range(preferences.num_clients):
                p_ds = partitioner.load_partition(i)
                df_p = p_ds.to_pandas()

                # Create a counter for this client: Key=(target, sensitive)
                client_groups = list(
                    zip(
                        df_p[preferences.target_attribute],
                        df_p[preferences.sensitive_attribute],
                        strict=False,
                    )
                )
                counter_groups.append(Counter(client_groups))

            # Plot
            title = "Samples for each group (target, sensitive) per client"
            plot_fairness_distributions(
                title,
                counter_groups,
                unique_groups,
                f"fairness_group_distribution_{preferences.partitioner_type}.png",
            )

            import matplotlib.pyplot as plt  # noqa: PLC0415
            import pandas as pd  # noqa: PLC0415
            import seaborn as sns  # noqa: PLC0415

            def compute_dataset_disparity(df, sensitive_col, target_col):
                # Disparity = P(y=1 | z=0) - P(y=1 | z=1)
                # Signed value indicates direction of unfairness.
                # Using Demographic Parity difference.

                # Calculate P(y=1 | z=0)
                df_z0 = df[df[sensitive_col] == 0]
                p_y1_z0 = df_z0[target_col].mean() if len(df_z0) > 0 else 0

                # Calculate P(y=1 | z=1)
                df_z1 = df[df[sensitive_col] == 1]
                p_y1_z1 = df_z1[target_col].mean() if len(df_z1) > 0 else 0

                return p_y1_z0 - p_y1_z1

            client_disparities = []
            client_ids = []
            client_types = []
            num_partitions = 150

            for i in range(num_partitions):
                p_ds = partitioner.load_partition(i)
                df_p = p_ds.to_pandas()

                disparity = compute_dataset_disparity(
                    df_p, "sex_binary", "occupation_binary"
                )
                client_disparities.append(disparity)
                client_ids.append(i)
                client_types.append(partitioner.client_types.get(i, "unknown"))

            # Create DataFrame for plotting
            disparity_df = pd.DataFrame(
                {
                    "client_id": client_ids,
                    "disparity": client_disparities,
                    "type": client_types,
                }
            )

            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=disparity_df, x="client_id", y="disparity", hue="type", dodge=False
            )
            plt.title("Signed Dataset Disparity (Demographic Parity) per Client")
            plt.ylabel("Disparity P(y=1|z=0) - P(y=1|z=1)")
            plt.axhline(0, color="black", linewidth=0.8)
            # Simplify x-axis labels if too many
            if num_partitions > 50:
                plt.xticks(
                    ticks=range(0, num_partitions, 10),
                    labels=[str(x) for x in range(0, num_partitions, 10)],
                )
            plt.savefig("disparity.png", bbox_inches="tight")
            plt.close()
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

parser.add_argument("--epsilon_lambda", type=float, default=None)
parser.add_argument("--epsilon_statistics", type=float, default=None)

parser.add_argument(
    "--num_client_cpus", type=float, default=1.0
)  # Percentage of CPUs used by each client
parser.add_argument(
    "--num_client_gpus", type=float, default=0.0
)  # Percentage of GPUs used by each client
parser.add_argument("--ray_num_cpus", type=int, default=40)
parser.add_argument("--ray_num_gpus", type=int, default=1)


parser.add_argument("--sensitive_attribute", type=str, default=None)
parser.add_argument("--target_attribute", type=str, default=None)

# Fairness Partitioner parameters
parser.add_argument("--ratio_unfair_clients", type=float, default=None)
parser.add_argument("--group_to_reduce", type=int, nargs="+", default=None)
parser.add_argument("--group_to_increment", type=int, nargs="+", default=None)
parser.add_argument("--ratio_unfairness", type=float, nargs="+", default=None)
parser.add_argument("--samples_per_client", type=int, default=None)
parser.add_argument(
    "--distribution_mode",
    type=str,
    default="per_group",
    choices=["per_group", "representative"],
    help="Distribution mode: 'per_group' for deterministic per-group allocation, "
         "'representative' for random sampling (matches old implementation)",
)


# Initialize global variables for client_fn/server_fn access
preferences: Preferences | None = None
partitioner: Any = None
client_manager: SimpleClientManager | None = None
wandb_run: Any = None


def main():
    """Execute the Federated Learning simulation."""
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

    client_resources = {
        "num_cpus": args.num_client_cpus,
        "num_gpus": args.num_client_gpus,
    }

    ray_num_cpus = args.ray_num_cpus
    ray_num_gpus = args.ray_num_gpus

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
        # Fairness args
        sensitive_attribute=args.sensitive_attribute,
        target_attribute=args.target_attribute,
        ratio_unfair_clients=args.ratio_unfair_clients,
        group_to_reduce=tuple(args.group_to_reduce) if args.group_to_reduce else None,
        group_to_increment=tuple(args.group_to_increment)
        if args.group_to_increment
        else None,
        ratio_unfairness=tuple(args.ratio_unfairness)
        if args.ratio_unfairness
        else None,
        samples_per_client=args.samples_per_client,
        distribution_mode=args.distribution_mode,
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
        epsilon_statistics=args.epsilon_statistics,
        epsilon_lambda=args.epsilon_lambda,
    )

    avg_probs_path = os.path.join(preferences.fed_dir, "avg_proba.pkl")
    if os.path.exists(avg_probs_path):
        os.remove(avg_probs_path)
        print(f"Removed {avg_probs_path}")

    if args.dataset_name == "dutch":
        preferences.model = "LinearClassificationNet"
    elif args.dataset_name == "abalone":
        preferences.model = "AbaloneNet"
    elif args.dataset_name == "mnist":
        preferences.model = "SimpleMNISTModel"
    elif args.dataset_name == "celeba":
        preferences.model = "CelebaNet"
    else:
        msg = f"Unknown dataset for model selection: {preferences.dataset_name}"
        raise ValueError(msg)

    # Populate model info from dataset if not provided
    if preferences.dataset_name:
        model_info = get_model_info_from_dataset(preferences.dataset_name)
        if preferences.num_classes is None:
            preferences.num_classes = model_info.get("num_classes")
        if preferences.in_channels is None:
            preferences.in_channels = model_info.get("in_channels")

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
    # Create your ServerApp
    global partitioner  # noqa: PLW0603
    partitioner = prepare_data(preferences=preferences)

    global client_manager  # noqa: PLW0603
    if preferences.partitioner_type == "fairness" and partitioner:
        client_manager = FairnessClientManager(
            preferences=preferences, client_types=partitioner.client_types
        )
    else:
        client_manager = SimpleClientManager(preferences=preferences)

    # Create your ServerApp
    server_app = ServerApp(server_fn=server_fn)

    # Concstruct the ClientApp passing the client generation function
    client_app = ClientApp(client_fn=client_fn)

    ram_memory = 16_000 * 1024 * 1024 * 2

    # (optional) specify Ray config
    ray_init_args = {
        "include_dashboard": False,
        "num_cpus": ray_num_cpus,
        "num_gpus": ray_num_gpus,
        "_memory": ram_memory,
        # "_redis_max_memory": 10000000,
        "object_store_memory": 78643200,
        "logging_level": logging.ERROR,
        "log_to_driver": True,
    }

    client_resources = {
        "num_cpus": args.num_client_cpus,
        "num_gpus": args.num_client_gpus,
    }

    configuration = {
        "client_resources": client_resources,
        "init_args": ray_init_args,
    }

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_clients,
        backend_config=configuration,
    )

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
