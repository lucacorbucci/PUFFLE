import argparse
import logging
import os
import shutil
import signal
from typing import Any

# Import from FlowerFLTemplate
from FlowerFLTemplate.ClientManager.client_manager import SimpleClientManager
from FlowerFLTemplate.ClientManager.fairness_client_manager import FairnessClientManager
from FlowerFLTemplate.main import prepare_data, setup_wandb
from FlowerFLTemplate.Models.utils import get_model
from FlowerFLTemplate.Server.server import Server
from FlowerFLTemplate.Utils.preferences import Preferences
from FlowerFLTemplate.Utils.utils import get_params, seed_everything
from flwr.client import ClientApp
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.simulation import run_simulation

# Import MMD-Fair components
from competitors.mmd_fair.simulation.client import MMDFairFlowerClient
from competitors.mmd_fair.simulation.metrics import (
    aggregate_evaluate_metrics,
    aggregate_fit_metrics,
)
from competitors.mmd_fair.simulation.strategy import MMDFairFedAvg

# Hide Ray logs
os.environ["RAY_LOG_LEVEL"] = "ERROR"

# Global variables for client_fn/server_fn access
preferences: Preferences | None = None
partitioner: Any = None
client_manager: SimpleClientManager | None = None
wandb_run: Any = None


def signal_handler(sig: int, frame: Any) -> None:
    """Handle interrupt signals gracefully."""
    print("Gracefully stopping experiment!")
    if wandb_run:
        wandb_run.finish()
    os._exit(0)


def client_fn(context: Context) -> Any:
    """
    Generate MMD-Fair Flower client instance.

    Args:
        context: Flower context with partition ID

    Returns:
        MMDFairFlowerClient instance

    """
    partition_id = int(context.node_config["partition-id"])

    if preferences is None:
        msg = "Preferences not initialized"
        raise ValueError(msg)

    # Capture preferences in closure to fix type narrowing
    prefs = preferences

    # Use lazy loading pattern
    def create_data_loader_fn():
        def load_data():
            partition = (
                partitioner.load_partition(partition_id) if partitioner else None
            )

            # Reuse FlowerFLTemplate's data loading
            from FlowerFLTemplate.Datasets.dataset_utils import (
                _create_abalone_dataloaders,
                _create_celeba_dataloaders,
                _create_dutch_dataloaders,
                prepare_mnist,
            )

            if prefs.dataset_name == "dutch":
                return _create_dutch_dataloaders(partition, prefs)
            if prefs.dataset_name == "mnist":
                trainloader = prepare_mnist(partition, prefs)
                return trainloader, trainloader
            if prefs.dataset_name == "abalone":
                return _create_abalone_dataloaders(partition, prefs)
            if prefs.dataset_name == "celeba":
                return _create_celeba_dataloaders(partition, prefs)
            msg = f"Unsupported dataset: {prefs.dataset_name}"
            raise ValueError(msg)

        return load_data

    return MMDFairFlowerClient(
        partition_id=partition_id,
        preferences=prefs,
        data_loader_fn=create_data_loader_fn(),
    ).to_client()


def server_fn(context: Context) -> ServerAppComponents:
    """
    Construct ServerAppComponents for MMD-Fair simulation.

    Args:
        context: Flower context

    Returns:
        ServerAppComponents with MMD-Fair strategy

    """
    if preferences is None:
        msg = "Preferences not initialized"
        raise ValueError(msg)

    # Create model
    if preferences.model is None:
        msg = "Model name not set"
        raise ValueError(msg)

    model = get_model(
        model_name=preferences.model,
        num_classes=preferences.num_classes,
        in_channels=preferences.in_channels,
    )

    ndarrays = get_params(model)
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Create MMD-Fair strategy
    # Get mu/ny from dynamic attributes (set in main())
    mu_val = float(getattr(preferences, "mu", 1.0))
    ny_val = int(getattr(preferences, "ny", 100))

    strategy = MMDFairFedAvg(
        fraction_fit=preferences.sampled_training_nodes_per_round or 1.0,
        fraction_evaluate=preferences.sampled_validation_nodes_per_round or 0.0,
        initial_parameters=global_model_init,
        preferences=preferences,
        wandb_run=wandb_run,
        mu=mu_val,
        ny=ny_val,
        lambda_fairness=preferences.regularization_lambda,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_evaluate_metrics,
    )

    config = ServerConfig(num_rounds=preferences.num_rounds or 1)

    if client_manager is None:
        msg = "Client Manager not initialized"
        raise ValueError(msg)

    server = Server(
        client_manager=client_manager, strategy=strategy, preferences=preferences
    )

    return ServerAppComponents(server=server, config=config)  # type: ignore


def main():
    """Execute MMD-Fair FL simulation."""
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser(description="MMD-Fair FL Simulation")

    # Reuse FlowerFLTemplate args
    parser.add_argument("--num_clients", type=int, required=True)
    parser.add_argument("--num_rounds", type=int, required=True)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partitioner_type", type=str, default="iid")
    parser.add_argument("--partitioner_alpha", type=float, default=None)
    parser.add_argument("--partitioner_by", type=str, default=None)
    parser.add_argument("--fed_dir", type=str, required=True)
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--project_name", type=str, default="MMDFair")
    parser.add_argument("--run_name", type=str, default=None)

    # MMD-Fair specific args
    parser.add_argument("--lambda_fairness", type=float, default=1.0)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--ny", type=int, default=100)

    # Fairness partitioner args
    parser.add_argument("--sensitive_attribute", type=str, default=None)
    parser.add_argument("--target_attribute", type=str, default=None)
    parser.add_argument("--ratio_unfair_clients", type=float, default=None)
    parser.add_argument("--group_to_reduce", type=int, nargs="+", default=None)
    parser.add_argument("--group_to_increment", type=int, nargs="+", default=None)
    parser.add_argument("--ratio_unfairness", type=float, nargs="+", default=None)
    parser.add_argument("--samples_per_client", type=int, default=None)
    parser.add_argument("--distribution_mode", type=str, default="per_group")

    # Resource args
    parser.add_argument("--num_client_cpus", type=float, default=1.0)
    parser.add_argument("--num_client_gpus", type=float, default=0.0)
    parser.add_argument("--ray_num_cpus", type=int, default=40)
    parser.add_argument("--ray_num_gpus", type=int, default=1)

    parser.add_argument("--num_test_nodes", type=int, default=None)
    parser.add_argument("--num_validation_nodes", type=int, default=None)
    parser.add_argument("--num_train_nodes", type=int, default=None)
    parser.add_argument(
        "--sampled_validation_nodes_per_round", type=float, default=None
    )
    parser.add_argument("--sampled_train_nodes_per_round", type=float, default=None)
    parser.add_argument("--sampled_test_nodes_per_round", type=float, default=None)

    args = parser.parse_args()

    seed_everything(args.seed)

    # Global preferences
    global preferences  # noqa: PLW0603
    preferences = Preferences(
        num_clients=args.num_clients,
        num_rounds=args.num_rounds,
        cross_device=True,  # Always cross-device for MMD-Fair
        num_epochs=args.num_epochs,
        num_test_nodes=args.num_test_nodes,
        num_validation_nodes=args.num_validation_nodes,
        num_train_nodes=args.num_train_nodes,
        sampled_validation_nodes_per_round=args.sampled_validation_nodes_per_round,
        sampled_training_nodes_per_round=args.sampled_train_nodes_per_round,
        sampled_test_nodes_per_round=args.sampled_test_nodes_per_round,
        seed=args.seed,
        fed_dir=args.fed_dir,
        fl_setting="cross_device",
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        partitioner_type=args.partitioner_type,
        partitioner_alpha=args.partitioner_alpha,
        partitioner_by=args.partitioner_by,
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
        optimizer="sgd",
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        regularization_lambda=args.lambda_fairness,
    )

    # Add MMD-Fair specific attributes (dynamic attributes for type checker)
    preferences.mu = args.mu  # type: ignore[attr-defined]
    preferences.ny = args.ny  # type: ignore[attr-defined]

    # Set model based on dataset
    if args.dataset_name == "dutch":
        preferences.model = "LinearClassificationNet"
    elif args.dataset_name == "abalone":
        preferences.model = "AbaloneNet"
    elif args.dataset_name == "mnist":
        preferences.model = "SimpleMNISTModel"
    elif args.dataset_name == "celeba":
        preferences.model = "CelebaNet"
    else:
        msg = f"Unknown dataset: {args.dataset_name}"
        raise ValueError(msg)

    # Populate model info
    from FlowerFLTemplate.Datasets.dataset_utils import get_model_info_from_dataset

    model_info = get_model_info_from_dataset(args.dataset_name)
    preferences.num_classes = model_info.get("num_classes")
    preferences.in_channels = model_info.get("in_channels")

    # Clean fed_dir
    if os.path.exists(args.fed_dir):
        for item in os.listdir(args.fed_dir):
            item_path = os.path.join(args.fed_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        os.makedirs(args.fed_dir, exist_ok=True)

    # Setup wandb
    global wandb_run  # noqa: PLW0603
    wandb_run = setup_wandb(args.project_name, args.run_name) if args.wandb else None

    # Prepare data
    global partitioner  # noqa: PLW0603
    partitioner = prepare_data(preferences=preferences)

    # Create client manager
    global client_manager  # noqa: PLW0603
    if preferences.partitioner_type == "fairness" and partitioner:
        client_manager = FairnessClientManager(
            preferences=preferences, client_types=partitioner.client_types
        )
    else:
        client_manager = SimpleClientManager(preferences=preferences)

    # Create apps
    server_app = ServerApp(server_fn=server_fn)
    client_app = ClientApp(client_fn=client_fn)

    # Ray config
    ray_init_args = {
        "include_dashboard": False,
        "num_cpus": args.ray_num_cpus,
        "num_gpus": args.ray_num_gpus,
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

    # Run simulation
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=args.num_clients,
        backend_config=configuration,
    )

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
