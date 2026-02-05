import os
from logging import INFO
from typing import Any

import dill
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
        Aggregates (evaluation) metrics from multiple clients using weighted averages.

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

        # Compute disparity with statistics from aggregated counters
        # This is the true disparity computed by aggregating local prediction counters
        has_counters = any("counter_z" in m for _, m in metrics)
        if has_counters:
            counter_z = sum(m.get("counter_z", 0) for _, m in metrics)
            counter_not_z = sum(m.get("counter_not_z", 0) for _, m in metrics)
            counter_y_z = sum(m.get("counter_y_z", 0) for _, m in metrics)
            counter_y_not_z = sum(m.get("counter_y_not_z", 0) for _, m in metrics)
            # log all the counters to wandb for each client
            for _, metric in metrics:
                if "counter_z" in metric:
                    wandb_run.log(
                        {f"counter_z_{metric['client_id']}": metric["counter_z"]}
                    )
                if "counter_not_z" in metric:
                    wandb_run.log(
                        {
                            f"counter_not_z_{metric['client_id']}": metric[
                                "counter_not_z"
                            ]
                        }
                    )
                if "counter_y_z" in metric:
                    wandb_run.log(
                        {f"counter_y_z_{metric['client_id']}": metric["counter_y_z"]}
                    )
                if "counter_y_not_z" in metric:
                    wandb_run.log(
                        {
                            f"counter_y_not_z_{metric['client_id']}": metric[
                                "counter_y_not_z"
                            ]
                        }
                    )

                if "counter_not_y_z" in metric:
                    wandb_run.log(
                        {
                            f"counter_not_y_z_{metric['client_id']}": metric[
                                "counter_not_y_z"
                            ]
                        }
                    )
                if "counter_not_y_not_z" in metric:
                    wandb_run.log(
                        {
                            f"counter_not_y_not_z_{metric['client_id']}": metric[
                                "counter_not_y_not_z"
                            ]
                        }
                    )
                if "counter_y" in metric:
                    wandb_run.log(
                        {f"counter_y_{metric['client_id']}": metric["counter_y"]}
                    )
                if "counter_not_y" in metric:
                    wandb_run.log(
                        {
                            f"counter_not_y_{metric['client_id']}": metric[
                                "counter_not_y"
                            ]
                        }
                    )
                if "total_samples" in metric:
                    wandb_run.log(
                        {
                            f"total_samples_{metric['client_id']}": metric[
                                "total_samples"
                            ]
                        }
                    )

                disparity_client = abs(
                    metric["counter_y_z"] / metric["counter_z"]
                    - metric["counter_y_not_z"] / metric["counter_not_z"]
                )
                wandb_run.log({f"disparity_{metric['client_id']}": disparity_client})

            # Compute P(Y=1|Z=1) and P(Y=1|Z=0)
            p_y_given_z = counter_y_z / counter_z if counter_z > 0 else 0
            p_y_given_not_z = (
                counter_y_not_z / counter_not_z if counter_not_z > 0 else 0
            )
            disparity_with_statistics = abs(p_y_given_z - p_y_given_not_z)

            log(
                INFO,
                f"{mode.name.title()} Disparity with statistics: {disparity_with_statistics:.4f} - "
                f"Counter Z: {counter_z} - Counter Not Z: {counter_not_z} - "
                f"Counter Y|Z: {counter_y_z} - Counter Y|Not Z: {counter_y_not_z}",
            )

            agg_metrics[f"{mode.name.title()} Disparity with statistics"] = (
                disparity_with_statistics
            )

        # Compute Dataset Disparity (Ground Truth)
        has_dataset_counters = any("dataset_counter_z" in m for _, m in metrics)
        if has_dataset_counters:
            d_counter_z = sum(m.get("dataset_counter_z", 0) for _, m in metrics)
            d_counter_not_z = sum(m.get("dataset_counter_not_z", 0) for _, m in metrics)
            d_counter_y_z = sum(m.get("dataset_counter_y_z", 0) for _, m in metrics)
            d_counter_y_not_z = sum(
                m.get("dataset_counter_y_not_z", 0) for _, m in metrics
            )

            # Compute P(Y=1|Z=1) and P(Y=1|Z=0) for Dataset
            d_p_y_given_z = d_counter_y_z / d_counter_z if d_counter_z > 0 else 0
            d_p_y_given_not_z = (
                d_counter_y_not_z / d_counter_not_z if d_counter_not_z > 0 else 0
            )
            dataset_disparity = abs(d_p_y_given_z - d_p_y_given_not_z)

            agg_metrics[f"{mode.name.title()} Dataset Disparity"] = dataset_disparity

            # Log dataset counters to wandb
            if wandb_run:
                wandb_run.log(
                    {
                        f"{mode.name.title()}_Dataset_Disparity": dataset_disparity,
                        f"{mode.name.title()}_Dataset_Counter_Z": d_counter_z,
                        f"{mode.name.title()}_Dataset_Counter_Y_Z": d_counter_y_z,
                    }
                )

                # Log dataset disparity per client
                for _, metric in metrics:
                    if "dataset_counter_z" in metric:
                        d_c_z = metric.get("dataset_counter_z", 0)
                        d_c_not_z = metric.get("dataset_counter_not_z", 0)
                        d_c_y_z = metric.get("dataset_counter_y_z", 0)
                        d_c_y_not_z = metric.get("dataset_counter_y_not_z", 0)

                        p_y_z = d_c_y_z / d_c_z if d_c_z > 0 else 0
                        p_y_not_z = d_c_y_not_z / d_c_not_z if d_c_not_z > 0 else 0

                        d_disp_client = abs(p_y_z - p_y_not_z)
                        wandb_run.log(
                            {
                                f"dataset_disparity_{metric.get('client_id', 'unknown')}": d_disp_client
                            }
                        )

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
                f"Train Disparity with statistics: {counter_disparity:.4f} - "
                f"Counter Z: {counter_z} - Counter Not Z: {counter_not_z} - "
                f"Counter Y|Z: {counter_y_z} - Counter Y|Not Z: {counter_y_not_z}",
            )

            agg_metrics["Train Disparity with statistics"] = counter_disparity

        # Handle DP statistics aggregation
        has_noisy_counters = any("counter_y_z_noise" in m for _, m in metrics)
        if has_noisy_counters:
            sum_y_z_noise = sum(m.get("counter_y_z_noise", 0) for _, m in metrics)
            sum_y_not_z_noise = sum(
                m.get("counter_y_not_z_noise", 0) for _, m in metrics
            )
            sum_z = sum(m.get("counter_z", 0) for _, m in metrics)
            sum_not_z = sum(m.get("counter_not_z", 0) for _, m in metrics)

            avg_probs = {}
            # Assuming binary Z/Y where Y=1|Z=1 maps to "1|1" and Y=1|Z=0 maps to "1|0"
            if sum_z > 0:
                p_1_1 = sum_y_z_noise / sum_z
                avg_probs["1|1"] = float(max(0.0, min(1.0, p_1_1)))
            else:
                avg_probs["1|1"] = 0.0

            if sum_not_z > 0:
                p_1_0 = sum_y_not_z_noise / sum_not_z
                avg_probs["1|0"] = float(max(0.0, min(1.0, p_1_0)))
            else:
                avg_probs["1|0"] = 0.0

            # Save to disk
            if fed_dir:
                try:
                    with open(os.path.join(fed_dir, "avg_proba.pkl"), "wb") as f:
                        dill.dump(avg_probs, f)
                except Exception as e:  # noqa: BLE001
                    log(INFO, f"Failed to save average probabilities: {e}")

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
