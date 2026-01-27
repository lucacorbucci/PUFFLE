from logging import INFO
from typing import Any

from flwr.common.logger import log


class Aggregation:
    @staticmethod
    def agg_metrics_test(metrics: list, server_round: int, wandb_run: Any) -> dict:
        """
        Aggregates test metrics from multiple clients using weighted averages.

        Supports classification (accuracy, loss) and regression (rmse, mae, r2, mse, loss) metrics.
        Logs aggregated values and updates wandb run if provided.

        Args:
            metrics (list): List of tuples (num_examples, metric_dict) from clients.
            server_round (int): Current federated learning round.
            wandb_run (Any): Weights & Biases run instance for logging.

        Returns:
            dict: Aggregated metrics dictionary with keys like "Test Loss", "Test_Accuracy" or regression equivalents, and "FL Round".

        """
        total_examples = sum([n_examples for n_examples, _ in metrics])

        loss_test = (
            sum([n_examples * metric["loss"] for n_examples, metric in metrics])
            / total_examples
        )

        agg_metrics = {}

        if metrics[0][1].get("accuracy"):
            accuracy_test = (
                sum([n_examples * metric["accuracy"] for n_examples, metric in metrics])
                / total_examples
            )
            log(
                INFO,
                f"Test Accuracy: {accuracy_test} - Test Loss {loss_test}",
            )

            agg_metrics["Test Loss"] = loss_test
            agg_metrics["Test_Accuracy"] = accuracy_test
            agg_metrics["FL Round"] = server_round

        if metrics[0][1].get("rmse"):
            rmse_test = (
                sum([n_examples * metric["rmse"] for n_examples, metric in metrics])
                / total_examples
            )
            mae_test = (
                sum([n_examples * metric["mae"] for n_examples, metric in metrics])
                / total_examples
            )
            r2_test = (
                sum([n_examples * metric["r2"] for n_examples, metric in metrics])
                / total_examples
            )
            mse_test = (
                sum([n_examples * metric["mse"] for n_examples, metric in metrics])
                / total_examples
            )

            agg_metrics["Test Loss"] = loss_test
            agg_metrics["rmse_test"] = rmse_test
            agg_metrics["mae_test"] = mae_test
            agg_metrics["r2_test"] = r2_test
            agg_metrics["mse_test"] = mse_test
            agg_metrics["FL Round"] = server_round

        if wandb_run:
            wandb_run.log(agg_metrics)

        return agg_metrics

    @staticmethod
    def agg_metrics_evaluation(
        metrics: list, server_round: int, wandb_run: Any
    ) -> dict:
        """
        Aggregates validation (evaluation) metrics from multiple clients using weighted averages.

        Supports classification (accuracy, loss) and regression (rmse, mae, r2, mse, loss) metrics.
        Logs aggregated values and updates wandb run if provided.

        Args:
            metrics (list): List of tuples (num_examples, metric_dict) from clients.
            server_round (int): Current federated learning round.
            wandb_run (Any): Weights & Biases run instance for logging.

        Returns:
            dict: Aggregated metrics dictionary with keys like "Validation Loss", "Validation_Accuracy" or regression equivalents, and "FL Round".

        """
        total_examples = sum([n_examples for n_examples, _ in metrics])
        agg_metrics = {}
        loss_evaluation = (
            sum([n_examples * metric["loss"] for n_examples, metric in metrics])
            / total_examples
        )
        if metrics[0][1].get("accuracy"):
            accuracy_evaluation = (
                sum([n_examples * metric["accuracy"] for n_examples, metric in metrics])
                / total_examples
            )

            agg_metrics["Validation Loss"] = loss_evaluation
            agg_metrics["Validation_Accuracy"] = accuracy_evaluation
            agg_metrics["FL Round"] = server_round
        if metrics[0][1].get("rmse"):
            rmse_evaluation = (
                sum([n_examples * metric["rmse"] for n_examples, metric in metrics])
                / total_examples
            )
            mae_evaluation = (
                sum([n_examples * metric["mae"] for n_examples, metric in metrics])
                / total_examples
            )
            r2_evaluation = (
                sum([n_examples * metric["r2"] for n_examples, metric in metrics])
                / total_examples
            )
            mse_evaluation = (
                sum([n_examples * metric["mse"] for n_examples, metric in metrics])
                / total_examples
            )

            agg_metrics["Validation Loss"] = loss_evaluation
            agg_metrics["rmse_evaluation"] = rmse_evaluation
            agg_metrics["mae_evaluation"] = mae_evaluation
            agg_metrics["r2_evaluation"] = r2_evaluation
            agg_metrics["mse_evaluation"] = mse_evaluation
            agg_metrics["FL Round"] = server_round

        if wandb_run:
            wandb_run.log(agg_metrics)

        return agg_metrics

    @staticmethod
    def agg_metrics_train(
        metrics: list, server_round: int, fed_dir: Any, wandb_run: Any
    ) -> dict:
        """
        Aggregates training metrics from multiple clients using weighted averages.

        Handles loss always; accuracy if present in metrics. Logs aggregated values and updates wandb run if provided.
        Note: fed_dir parameter is unused in the implementation.

        Args:
            metrics (list): List of tuples (num_examples, metric_dict) from clients.
            server_round (int): Current federated learning round.
            fed_dir (Any): Federated directory (unused).
            wandb_run (Any): Weights & Biases run instance for logging.

        Returns:
            dict: Aggregated metrics dictionary with "Train Loss", optional "Train Accuracy", and "FL Round".

        """
        # Collect the losses logged during each epoch in each client
        total_examples = sum([n_examples for n_examples, _ in metrics])
        losses = []
        accuracies = []
        accuracy_log = False
        statistics = []
        for n_examples, node_metrics in metrics:
            losses.append(n_examples * node_metrics["loss"])
            if node_metrics.get("accuracy"):
                accuracies.append(n_examples * node_metrics["accuracy"])
                accuracy_log = True

            # Create the dictionary we want to log. For some metrics we want to log
            # we have to check if they are present or not.
            to_be_logged = {
                "FL Round": server_round,
            }
            statistics.append(node_metrics.get("statistics", []))
            if wandb_run:
                wandb_run.log(
                    to_be_logged,
                )

        if accuracy_log:
            log(
                INFO,
                f"Train Accuracy: {sum(accuracies) / total_examples} - Train Loss {sum(losses) / total_examples}",
            )

            agg_metrics = {
                "Train Loss": sum(losses) / total_examples,
                "Train Accuracy": sum(accuracies) / total_examples,
                "FL Round": server_round,
            }
        else:
            log(
                INFO,
                f"Train Loss {sum(losses) / total_examples}",
            )

            agg_metrics = {
                "Train Loss": sum(losses) / total_examples,
                "FL Round": server_round,
            }

        # Handle fairness counters
        # PUFFLEModel returns flat keys: counter_z, counter_not_z, counter_y_z, counter_y_not_z
        # We need to sum them up across all clients.
        # Note: These are totals from the clients if they are properly accumulated or last batch stats.
        # Assuming PUFFLEModel returns counts from the last epoch/batch accumulation logic.
        # Check if any client returned counters
        has_counters = any("counter_z" in m for _, m in metrics)

        if has_counters:
            counter_z = sum([m.get("counter_z", 0) for _, m in metrics])
            counter_not_z = sum([m.get("counter_not_z", 0) for _, m in metrics])
            counter_y_z = sum([m.get("counter_y_z", 0) for _, m in metrics])
            counter_y_not_z = sum([m.get("counter_y_not_z", 0) for _, m in metrics])

            first_part = counter_y_z / counter_z if counter_z > 0 else 0
            second_part = counter_y_not_z / counter_not_z if counter_not_z > 0 else 0
            disparity = abs(first_part - second_part)
            log(
                INFO,
                f"Disparity: {disparity} - Counter Z: {counter_z} - Counter Not Z: {counter_not_z} - Counter Y Z: {counter_y_z} - Counter Y Not Z: {counter_y_not_z}",
            )

            agg_metrics["Disparity"] = disparity

        if wandb_run:
            wandb_run.log(agg_metrics)

        return agg_metrics
