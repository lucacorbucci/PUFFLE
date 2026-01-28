# ABOUTME: Aggregation functions for FL metrics from multiple clients.
# ABOUTME: Handles train, validation, and test metrics with fairness support.

from logging import INFO
from typing import Any

from flwr.common.logger import log

from puffle.Utils.modes import MetricMode


class Aggregation:
    @staticmethod
    def agg_metrics_test(
        metrics: list,
        server_round: int,
        wandb_run: Any,
    ) -> dict:
        """
        Aggregates test metrics from multiple clients using weighted averages.

        Supports classification (accuracy, loss, f1, disparity) metrics.
        Logs aggregated values and updates wandb run if provided.

        Args:
            metrics (list): List of tuples (num_examples, metric_dict) from clients.
            server_round (int): Current federated learning round.
            wandb_run (Any): Weights & Biases run instance for logging.

        Returns:
            dict: Aggregated metrics dictionary.

        """
        mode = MetricMode.TEST
        return Aggregation._aggregate_evaluation_metrics(
            metrics=metrics,
            server_round=server_round,
            wandb_run=wandb_run,
            mode=mode,
        )

    @staticmethod
    def agg_metrics_evaluation(
        metrics: list,
        server_round: int,
        wandb_run: Any,
    ) -> dict:
        """
        Aggregates validation (evaluation) metrics from multiple clients using weighted averages.

        Supports classification (accuracy, loss, f1, disparity) metrics.
        Logs aggregated values and updates wandb run if provided.

        Args:
            metrics (list): List of tuples (num_examples, metric_dict) from clients.
            server_round (int): Current federated learning round.
            wandb_run (Any): Weights & Biases run instance for logging.

        Returns:
            dict: Aggregated metrics dictionary.

        """
        mode = MetricMode.VALIDATION
        return Aggregation._aggregate_evaluation_metrics(
            metrics=metrics,
            server_round=server_round,
            wandb_run=wandb_run,
            mode=mode,
        )

    @staticmethod
    def _aggregate_evaluation_metrics(
        metrics: list,
        server_round: int,
        wandb_run: Any,
        mode: MetricMode,
    ) -> dict:
        """
        Internal helper to aggregate evaluation/test metrics.

        Args:
            metrics (list): List of tuples (num_examples, metric_dict) from clients.
            server_round (int): Current federated learning round.
            wandb_run (Any): Weights & Biases run instance for logging.
            mode (MetricMode): The metric mode (VALIDATION or TEST).

        Returns:
            dict: Aggregated metrics dictionary.

        """
        total_examples = sum(n_examples for n_examples, _ in metrics)
        agg_metrics: dict[str, Any] = {"FL Round": server_round}

        # Get the prefix for the mode (e.g., "val" or "test")
        prefix = mode.value

        # Aggregate loss - try both prefixed and unprefixed keys
        loss_key = f"{prefix}_loss"
        loss_values = []
        for n_examples, metric in metrics:
            if loss_key in metric:
                loss_values.append(n_examples * metric[loss_key])
            elif "loss" in metric:
                loss_values.append(n_examples * metric["loss"])

        if loss_values:
            aggregated_loss = sum(loss_values) / total_examples
            agg_metrics[f"{mode.name.title()} Loss"] = aggregated_loss

        # Aggregate accuracy - try both prefixed and unprefixed keys
        accuracy_key = f"{prefix}_accuracy"
        accuracy_values = []
        for n_examples, metric in metrics:
            if accuracy_key in metric:
                accuracy_values.append(n_examples * metric[accuracy_key])
            elif "accuracy" in metric:
                accuracy_values.append(n_examples * metric["accuracy"])

        if accuracy_values:
            aggregated_accuracy = sum(accuracy_values) / total_examples
            agg_metrics[f"{mode.name.title()}_Accuracy"] = aggregated_accuracy

        # Aggregate f1 - try both prefixed and unprefixed keys
        f1_key = f"{prefix}_f1"
        f1_values = []
        for n_examples, metric in metrics:
            if f1_key in metric:
                f1_values.append(n_examples * metric[f1_key])
            elif "f1" in metric:
                f1_values.append(n_examples * metric["f1"])

        if f1_values:
            aggregated_f1 = sum(f1_values) / total_examples
            agg_metrics[f"{mode.name.title()}_F1"] = aggregated_f1

        # Aggregate disparity - try both prefixed and unprefixed keys
        disparity_key = f"{prefix}_disparity"
        disparity_values = []
        for n_examples, metric in metrics:
            if disparity_key in metric:
                disparity_values.append(n_examples * metric[disparity_key])
            elif "disparity" in metric:
                disparity_values.append(n_examples * metric["disparity"])

        if disparity_values:
            aggregated_disparity = sum(disparity_values) / total_examples
            agg_metrics[f"{mode.name.title()}_Disparity"] = aggregated_disparity

        # Log metrics
        if accuracy_values:
            log(
                INFO,
                f"{mode.name.title()} Accuracy: {agg_metrics.get(f'{mode.name.title()}_Accuracy', 0):.4f} - "
                f"{mode.name.title()} Loss: {agg_metrics.get(f'{mode.name.title()} Loss', 0):.4f}",
            )

        if wandb_run:
            wandb_run.log(agg_metrics)

        return agg_metrics

    @staticmethod
    def agg_metrics_train(
        metrics: list,
        server_round: int,
        wandb_run: Any,
        fed_dir: Any = None,
    ) -> dict:
        """
        Aggregates training metrics from multiple clients using weighted averages.

        Handles loss, accuracy, f1, disparity if present in metrics.
        Logs aggregated values and updates wandb run if provided.

        Args:
            metrics (list): List of tuples (num_examples, metric_dict) from clients.
            server_round (int): Current federated learning round.
            wandb_run (Any): Weights & Biases run instance for logging.
            fed_dir (Any): Federated directory (unused, kept for API compatibility).

        Returns:
            dict: Aggregated metrics dictionary with training metrics and "FL Round".

        """
        _ = fed_dir  # unused but kept for API compatibility

        total_examples = sum(n_examples for n_examples, _ in metrics)
        mode = MetricMode.TRAIN
        prefix = mode.value  # "train"

        agg_metrics: dict[str, Any] = {"FL Round": server_round}

        # Aggregate loss - try both prefixed and unprefixed keys
        loss_key = f"{prefix}_loss"
        loss_values = []
        for n_examples, metric in metrics:
            if loss_key in metric:
                loss_values.append(n_examples * metric[loss_key])
            elif "loss" in metric:
                loss_values.append(n_examples * metric["loss"])

        if loss_values:
            aggregated_loss = sum(loss_values) / total_examples
            agg_metrics["Train Loss"] = aggregated_loss

        # Aggregate accuracy - try both prefixed and unprefixed keys
        accuracy_key = f"{prefix}_accuracy"
        accuracy_values = []
        for n_examples, metric in metrics:
            if accuracy_key in metric:
                accuracy_values.append(n_examples * metric[accuracy_key])
            elif "accuracy" in metric:
                accuracy_values.append(n_examples * metric["accuracy"])

        if accuracy_values:
            aggregated_accuracy = sum(accuracy_values) / total_examples
            agg_metrics["Train Accuracy"] = aggregated_accuracy

        # Aggregate f1 - try both prefixed and unprefixed keys
        f1_key = f"{prefix}_f1"
        f1_values = []
        for n_examples, metric in metrics:
            if f1_key in metric:
                f1_values.append(n_examples * metric[f1_key])
            elif "f1" in metric:
                f1_values.append(n_examples * metric["f1"])

        if f1_values:
            aggregated_f1 = sum(f1_values) / total_examples
            agg_metrics["Train F1"] = aggregated_f1

        # Aggregate disparity - try both prefixed and unprefixed keys
        disparity_key = f"{prefix}_disparity"
        disparity_values = []
        for n_examples, metric in metrics:
            if disparity_key in metric:
                disparity_values.append(n_examples * metric[disparity_key])
            elif "disparity" in metric:
                disparity_values.append(n_examples * metric["disparity"])

        if disparity_values:
            aggregated_disparity = sum(disparity_values) / total_examples
            agg_metrics["Train Disparity"] = aggregated_disparity

        # Handle fairness counters from PUFFLEModel
        # PUFFLEModel returns: counter_z, counter_not_z, counter_y_z, counter_y_not_z
        has_counters = any("counter_z" in m for _, m in metrics)

        if has_counters:
            counter_z = sum(m.get("counter_z", 0) for _, m in metrics)
            counter_not_z = sum(m.get("counter_not_z", 0) for _, m in metrics)
            counter_y_z = sum(m.get("counter_y_z", 0) for _, m in metrics)
            counter_y_not_z = sum(m.get("counter_y_not_z", 0) for _, m in metrics)

            first_part = counter_y_z / counter_z if counter_z > 0 else 0
            second_part = counter_y_not_z / counter_not_z if counter_not_z > 0 else 0
            counter_disparity = abs(first_part - second_part)

            log(
                INFO,
                f"Counter Disparity: {counter_disparity:.4f} - "
                f"Counter Z: {counter_z} - Counter Not Z: {counter_not_z} - "
                f"Counter Y Z: {counter_y_z} - Counter Y Not Z: {counter_y_not_z}",
            )

            agg_metrics["Counter Disparity"] = counter_disparity

        # Log training metrics
        if accuracy_values:
            log(
                INFO,
                f"Train Accuracy: {agg_metrics.get('Train Accuracy', 0):.4f} - "
                f"Train Loss: {agg_metrics.get('Train Loss', 0):.4f}",
            )
        elif loss_values:
            log(INFO, f"Train Loss: {agg_metrics.get('Train Loss', 0):.4f}")

        if wandb_run:
            wandb_run.log(agg_metrics)

        return agg_metrics
