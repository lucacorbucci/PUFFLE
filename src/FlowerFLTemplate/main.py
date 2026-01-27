import argparse
from pathlib import Path

import flwr as fl
import wandb
from flwr.server.strategy import FedAvg

from FlowerFLTemplate.Aggregations.aggregations import Aggregation
from FlowerFLTemplate.Client.client import FlowerClient
from FlowerFLTemplate.Datasets.dataset_utils import load_partitioned_dataset
from FlowerFLTemplate.Utils.preferences import Preferences


def get_preferences() -> Preferences:
    """Parse command line arguments into Preferences object."""
    parser = argparse.ArgumentParser(description="Flower Simulation with PUFFLE")

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, default="./data/dutch.csv")
    parser.add_argument("--num_clients", type=int, required=True)
    parser.add_argument("--partitioner_type", type=str, default="iid")
    parser.add_argument("--partitioner_alpha", type=float, default=1.0)
    parser.add_argument("--partitioner_by", type=str, default="sex")

    # Simulation arguments
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--sampled_training_nodes_per_round", type=float, default=0.2)
    parser.add_argument("--sampled_validation_nodes_per_round", type=float, default=0.2)
    parser.add_argument("--sampled_test_nodes_per_round", type=float, default=0.0)
    parser.add_argument("--fed_dir", type=str, default="./fed_dir")
    parser.add_argument("--project_name", type=str, default="PUFFLE_FL")
    parser.add_argument("--run_name", type=str, default="test")
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=42)

    # PUFFLE arguments
    parser.add_argument("--unfairness_reduction", type=bool, default=False)
    parser.add_argument("--fairness_metric", type=str, default="disparity")
    parser.add_argument("--regularization_mode", type=str, default="fixed")
    parser.add_argument("--target", type=float, default=0.05)
    parser.add_argument("--regularization_lambda", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--weight_decay_alpha", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--task", type=str, default="classification")
    parser.add_argument("--fl_setting", type=str, default="cross_device")

    # Differential Privacy (optional)
    parser.add_argument("--private_training", type=bool, default=False)
    parser.add_argument("--epsilon", type=float, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Dummy args for sweep compatibility
    parser.add_argument("--num_train_nodes", type=int, default=None)
    parser.add_argument("--num_validation_nodes", type=int, default=None)
    parser.add_argument("--num_test_nodes", type=int, default=None)

    # Model architecture arguments
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--in_channels", type=int, default=None)
    parser.add_argument("--pixel", type=int, default=None)

    args = parser.parse_args()

    return Preferences(**vars(args))


def main():
    preferences = get_preferences()

    num_clients = preferences.num_clients
    num_rounds = preferences.num_rounds
    dataset_name = preferences.dataset_name
    dataset_path = preferences.dataset_path
    partitioner_type = preferences.partitioner_type
    partitioner_alpha = preferences.partitioner_alpha
    partitioner_by = preferences.partitioner_by
    seed = preferences.seed
    fed_dir = preferences.fed_dir

    if num_clients is None:
        msg = "num_clients must be set"
        raise ValueError(msg)
    if num_rounds is None:
        msg = "num_rounds must be set"
        raise ValueError(msg)
    if dataset_name is None:
        msg = "dataset_name must be set"
        raise ValueError(msg)
    if dataset_path is None:
        msg = "dataset_path must be set"
        raise ValueError(msg)
    if partitioner_type is None:
        msg = "partitioner_type must be set"
        raise ValueError(msg)
    if partitioner_alpha is None:
        msg = "partitioner_alpha must be set"
        raise ValueError(msg)
    if seed is None:
        msg = "seed must be set"
        raise ValueError(msg)
    if fed_dir is None:
        msg = "fed_dir must be set"
        raise ValueError(msg)

    # Initialize WandB
    if preferences.wandb:
        run = wandb.init(
            project=preferences.project_name,
            name=preferences.run_name,
            config=vars(preferences),
        )
    else:
        run = None

    # Load and partition data
    Path(fed_dir).mkdir(parents=True, exist_ok=True)

    partitions = load_partitioned_dataset(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        num_clients=num_clients,
        partitioner_type=partitioner_type,
        partitioner_alpha=partitioner_alpha,
        partitioner_by=partitioner_by,
        seed=seed,
        fed_dir=fed_dir,
    )

    def client_fn(cid: str) -> fl.client.Client:
        partition_id = int(cid)
        trainloader = partitions[partition_id]["train"]
        valloader = partitions[partition_id]["validation"]

        return FlowerClient(
            trainloader=trainloader,
            valloader=valloader,
            preferences=preferences,
            partition_id=partition_id,
        ).to_client()

    # Define strategy
    fraction_fit = preferences.sampled_training_nodes_per_round
    fraction_evaluate = preferences.sampled_validation_nodes_per_round

    if fraction_fit is None:
        msg = "sampled_training_nodes_per_round must be set"
        raise ValueError(msg)
    if fraction_evaluate is None:
        msg = "sampled_validation_nodes_per_round must be set"
        raise ValueError(msg)

    class PUFFLEStrategy(FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            if not results:
                return None, {}

            parameters_aggregated, metrics_aggregated = super().aggregate_fit(
                server_round, results, failures
            )

            metrics = [(r.num_examples, r.metrics) for _, r in results]

            custom_metrics = Aggregation.agg_metrics_train(
                metrics, server_round, preferences.fed_dir, run
            )

            metrics_aggregated.update(custom_metrics)

            return parameters_aggregated, metrics_aggregated

        def aggregate_evaluate(self, server_round, results, failures):
            if not results:
                return None, {}

            loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
                server_round, results, failures
            )

            metrics = [(r.num_examples, r.metrics) for _, r in results]
            custom_metrics = Aggregation.agg_metrics_evaluation(
                metrics, server_round, run
            )
            metrics_aggregated.update(custom_metrics)

            return loss_aggregated, metrics_aggregated

    strategy = PUFFLEStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=max(1, int(num_clients * fraction_fit)),
        min_evaluate_clients=max(1, int(num_clients * fraction_evaluate)),
        min_available_clients=num_clients,
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1.0, "num_gpus": 0.0},
    )


if __name__ == "__main__":
    main()
