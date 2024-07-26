import json
import os

import dill


class AggregationFunctions:
    def agg_metrics_test(
        metrics: list,
        server_round: int,
        train_parameters,
        wandb_run,
        args,
        fed_dir: str,
    ) -> dict:
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
                        (
                            "test_accuracy"
                            if not train_parameters.sweep
                            else "validation_accuracy"
                        )
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

        if args.metric == "disparity":
            # Log data from the different test clients:
            for _, metric in metrics:
                node_name = metric["cid"]
                disparity = metric[
                    (
                        "max_disparity_test"
                        if not train_parameters.sweep
                        else "max_disparity_validation"
                    )
                ]
                accuracy = metric[
                    (
                        "test_accuracy"
                        if not train_parameters.sweep
                        else "validation_accuracy"
                    )
                ]
                disparity_dataset = metric.get("max_disparity_dataset", 0)
                agg_metrics = {
                    f"Test Node {node_name} - Acc.": accuracy,
                    f"Test Node {node_name} - Disp.": disparity,
                    f"Test Node {node_name} - Disp. Dataset": disparity_dataset,
                    "FL Round": server_round,
                }
                if wandb_run:
                    wandb_run.log(agg_metrics)
            (
                sum_counters,
                sum_targets,
                average_probabilities,
                max_disparity_statistics,
                disparity_combinations,
            ) = AggregationFunctions.handle_counters(metrics, "counters", fed_dir)
            if wandb_run:
                for combination in disparity_combinations:
                    target, sensitive_value, disparity = combination
                    wandb_run.log(
                        {
                            "FL Round": server_round,
                            f"Test Disparity P({target}, {sensitive_value}) - P({target}, NOT {sensitive_value})": abs(
                                disparity
                            ),
                        }
                    )

        if args.metric == "disparity":
            agg_metrics = {
                "Test Loss": loss_test,
                "Test Accuracy": accuracy_test,
                "Test Disparity with statistics": max_disparity_statistics,
                "FL Round": server_round,
                "Test F1": f1_test,
            }

        if wandb_run:
            wandb_run.log(agg_metrics)
        return agg_metrics

    def agg_metrics_evaluation(
        metrics: list,
        server_round: int,
        train_parameters,
        wandb_run,
        args,
        fed_dir: str,
    ) -> dict:
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
                        (
                            "test_accuracy"
                            if not train_parameters.sweep
                            else "validation_accuracy"
                        )
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

        if args.metric == "disparity":
            (
                sum_counters,
                sum_targets,
                average_probabilities,
                max_disparity_statistics,
                disparity_combinations,
            ) = AggregationFunctions.handle_counters(metrics, "counters", fed_dir)
            if wandb_run:
                for combination in disparity_combinations:
                    target, sensitive_value, disparity = combination
                    wandb_run.log(
                        {
                            "FL Round": server_round,
                            f"Validation Disparity P({target}, {sensitive_value}) - P({target}, NOT {sensitive_value})": abs(
                                disparity
                            ),
                        }
                    )

        custom_metric = accuracy_evaluation
        if args.target:
            if args.metric == "disparity":
                distance = args.target - max_disparity_statistics

            if distance > 0:
                penalty = 0
            else:
                penalty = -float("inf")

            custom_metric = accuracy_evaluation + penalty

        if args.metric == "disparity":
            agg_metrics = {
                "Validation Loss": loss_evaluation,
                "Validation_Accuracy": accuracy_evaluation,
                "Validation Disparity with statistics": max_disparity_statistics,
                "Custom_metric": custom_metric,
                "FL Round": server_round,
                "Validation F1": f1_validation,
            }

        if wandb_run:
            wandb_run.log(agg_metrics)
        return agg_metrics

    def agg_metrics_train(
        metrics: list,
        server_round: int,
        current_max_epsilon: float,
        fed_dir,
        wandb_run=None,
        args=None,
    ) -> dict:
        losses = []
        losses_with_regularization = []
        epsilon_list = []
        accuracies = []
        lambda_list = []
        max_disparity_train = []

        total_examples = sum([n_examples for n_examples, _ in metrics])
        agg_metrics = {
            "FL Round": server_round,
        }

        # Generic statistics that are logged for each round
        # and that are not dependent on the metric we are using
        for n_examples, node_metrics in metrics:
            losses.append(n_examples * node_metrics["train_loss"])

            losses_with_regularization.append(
                n_examples * node_metrics["train_loss_with_regularization"]
            )
            epsilon_list.append(node_metrics["epsilon"])
            accuracies.append(n_examples * node_metrics["train_accuracy"])
            lambda_list.append(node_metrics["Lambda"])
            client_id = node_metrics["cid"]
            DPL_lambda = node_metrics["Lambda"]

            if DPL_lambda:
                agg_metrics[f"Lambda Client {client_id}"] = DPL_lambda

            if args.metric == "disparity":
                disparity_client_after_local_epoch = node_metrics["Disparity Train"]
                agg_metrics = {
                    f"Disparity Client {client_id} After Local train": disparity_client_after_local_epoch,
                }

        current_max_epsilon = max(current_max_epsilon, *epsilon_list)
        agg_metrics["Train Loss"] = sum(losses) / total_examples
        agg_metrics["Train Accuracy"] = sum(accuracies) / total_examples
        agg_metrics["Train Loss with Regularization"] = (
            sum(losses_with_regularization) / total_examples
        )
        agg_metrics["Aggregated Lambda"] = (
            sum(lambda_list) / len(lambda_list)
            if args.regularization_mode == "tunable"
            else args.regularization_lambda
        )
        agg_metrics["Train Epsilon"] = current_max_epsilon

        if wandb_run:
            wandb_run.log(
                agg_metrics,
            )

        # now we compute some other aggregated metrics on the entire
        # metrics list returned by the clients
        if args.metric == "disparity":
            (
                sum_counters,
                sum_targets,
                average_probabilities,
                max_disparity_statistics,
                _,
            ) = AggregationFunctions.handle_counters(metrics, "counters", fed_dir)
            with open(f"{fed_dir}/avg_proba.pkl", "wb") as file:
                dill.dump(average_probabilities, file)
            (
                sum_counters_no_noise,
                sum_targets_no_noise,
                _,
                max_disparity_statistics_no_noise,
                disparity_combinations_no_noise,
            ) = AggregationFunctions.handle_counters(
                metrics, "counters_no_noise", fed_dir
            )
            if wandb_run:
                for combination in disparity_combinations_no_noise:
                    target, sensitive_value, disparity = combination
                    wandb_run.log(
                        {
                            "FL Round": server_round,
                            f"Train Disparity P({target}, {sensitive_value}) - P({target}, NOT {sensitive_value})": abs(
                                disparity
                            ),
                        }
                    )

            wandb_run.log(
                {
                    "Training Disparity with statistics": max_disparity_statistics,
                    "Training Disparity with statistics no noise": max_disparity_statistics_no_noise,
                    "FL Round": server_round,
                    "Average Probabilities": average_probabilities,
                }
            )

        return agg_metrics

    def handle_counters(metrics, key, fed_dir):
        # open the metadata file and update the counters
        with open(f"{fed_dir}/metadata.json", "r") as infile:
            json_file = json.load(infile)

        combinations = json_file["combinations"]  # ["1|0", "1|1"]
        all_combinations = json_file["all_combinations"]  # ["0|0", "0|1", "1|0", "1|1"]
        missing_combinations = json_file[
            "missing_combinations"
        ]  # [("0|0", "1|0"), ("0|1", "1|1")]
        # sum_counters = {"0|0": 0, "0|1": 0, "1|0": 0, "1|1": 0}
        sum_counters = {key: 0 for key in all_combinations}
        possible_sensitive_attributes = json_file["possible_z"]
        possible_targets = json_file["possible_y"]

        sum_possible_sensitive_attributes = {
            key: 0 for key in possible_sensitive_attributes
        }  # {"0": 0, "1": 0}

        for _, metric in metrics:
            metric = metric[key]
            for combination in combinations:
                try:
                    sum_counters[combination] += metric[combination]
                except:
                    continue

            for sensitive_attribute in possible_sensitive_attributes:
                try:
                    sum_possible_sensitive_attributes[sensitive_attribute] += metric[
                        sensitive_attribute
                    ]
                except:
                    continue

        for non_existing, existing in missing_combinations:
            sum_counters[non_existing] = (
                sum_possible_sensitive_attributes[existing[-1]] - sum_counters[existing]
                if sum_possible_sensitive_attributes[existing[-1]]
                - sum_counters[existing]
                > 0
                else 0
            )
        average_probabilities = {}
        for combination in all_combinations:
            try:
                proba = (
                    sum_counters[combination]
                    / sum_possible_sensitive_attributes[combination[2]]
                )
                if proba > 1:
                    proba = 1
                if proba < 0:
                    proba = 0
                average_probabilities[combination] = proba
            except:
                continue

        max_disparity_statistics = []
        combinations_disparity = []
        for target in possible_targets:
            for sensitive_value in possible_sensitive_attributes:
                Y_target_Z_sensitive_value = sum_counters[f"{target}|{sensitive_value}"]
                Z_sensitive_value = sum_possible_sensitive_attributes[sensitive_value]
                Z_not_sensitive_value = 0
                Y_target_Z_not_sensitive_value = 0
                for not_sensitive_value in possible_sensitive_attributes:
                    if not_sensitive_value != sensitive_value:
                        Y_target_Z_not_sensitive_value += sum_counters[
                            f"{target}|{not_sensitive_value}"
                        ]
                        Z_not_sensitive_value += sum_possible_sensitive_attributes[
                            not_sensitive_value
                        ]

                disparity = abs(
                    Y_target_Z_sensitive_value / Z_sensitive_value
                    - Y_target_Z_not_sensitive_value / Z_not_sensitive_value
                )

                max_disparity_statistics.append(disparity)
                combinations_disparity.append((target, sensitive_value))

        max_disparity_with_statistics = max(max_disparity_statistics)
        if max_disparity_with_statistics < 0:
            max_disparity_with_statistics = 0
        if max_disparity_with_statistics > 1:
            max_disparity_with_statistics = 1

        combinations = [
            (target, sv, disparity)
            for (target, sv), disparity in zip(
                combinations_disparity, max_disparity_statistics
            )
        ]

        return (
            sum_counters,
            sum_possible_sensitive_attributes,
            average_probabilities,
            max_disparity_with_statistics,  # max_disparity_statistics,
            combinations,
        )
