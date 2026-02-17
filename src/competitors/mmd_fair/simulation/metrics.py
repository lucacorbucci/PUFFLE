from typing import List, Tuple

from flwr.common import Metrics


def weighted_average(metrics: List[Tuple[int, Metrics]], key: str) -> float:
    """Compute weighted average of a metric."""
    total_examples = sum(num_examples for num_examples, _ in metrics)
    weighted_sum = sum(
        num_examples * float(m.get(key, 0.0)) for num_examples, m in metrics
    )
    if total_examples == 0:
        return 0.0
    return weighted_sum / total_examples


def aggregate_fit_metrics(
    metrics: List[Tuple[int, Metrics]],
    server_round: int,
    wandb_run=None,
    fed_dir: str | None = None,
) -> Metrics:
    """
    Aggregate training metrics.

    Metrics to aggregate:
    - train_loss
    - train_accuracy
    - train_fairness (Unfairness / Demographic Disparity)
    - train_mmd_loss
    """
    if not metrics:
        return {}

    # Extract metrics
    train_loss = weighted_average(metrics, "train_loss")
    train_acc = weighted_average(metrics, "train_accuracy")
    train_fairness = weighted_average(metrics, "train_disparity")
    train_mmd = weighted_average(metrics, "train_mmd_loss")

    aggregated = {
        "train_loss": train_loss,
        "train_acc": train_acc,
        "train_fairness": train_fairness,
        "train_mmd_loss": train_mmd,
    }

    if wandb_run:
        wandb_run.log({"round": server_round, **aggregated})

    return aggregated


def aggregate_evaluate_metrics(
    metrics: List[Tuple[int, Metrics]],
    server_round: int,
    wandb_run=None,
    target: float | None = None,
) -> Metrics:
    """
    Aggregate evaluation metrics.

    Metrics to aggregate:
    - val_loss
    - val_acc
    - val_fairness
    - val_fair_fl_p1
    - val_mmd_loss
    """
    if not metrics:
        return {}

    val_loss = weighted_average(metrics, "loss")
    val_acc = weighted_average(metrics, "accuracy")
    val_fairness = weighted_average(metrics, "unfairness")
    val_p1 = weighted_average(metrics, "fair_fl_p1")
    val_mmd = weighted_average(metrics, "mmd_loss")

    aggregated = {
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_fairness": val_fairness,
        "val_fair_fl_p1": val_p1,
        "val_mmd_loss": val_mmd,
    }

    if wandb_run:
        wandb_run.log({"round": server_round, **aggregated})

    return aggregated
