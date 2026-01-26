import json
import os

import dill
import numpy as np
import torch


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
                    n_examples * metric["test_loss" if not train_parameters.sweep else "validation_loss"]
                    for n_examples, metric in metrics
                ]
            )
            / total_examples
        )
        accuracy_test = (
            sum(
                [
                    n_examples * metric["test_accuracy" if not train_parameters.sweep else "validation_accuracy"]
                    for n_examples, metric in metrics
                ]
            )
            / total_examples
        )
        f1_test = sum([n_examples * metric["f1_score"] for n_examples, metric in metrics]) / total_examples

        if args.metric == "disparity":
            # Log data from the different test clients:
            for _, metric in metrics:
                node_name = metric["cid"]
                disparity = metric["max_disparity_test" if not train_parameters.sweep else "max_disparity_validation"]
                accuracy = metric["test_accuracy" if not train_parameters.sweep else "validation_accuracy"]
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

            # metrics_fair = [
            #     (x, metric) for x, metric in metrics if int(metric["cid"]) < 75
            # ]
            # (
            #     _,
            #     _,
            #     _,
            #     max_disparity_statistics_fair,
            #     _,
            # ) = AggregationFunctions.handle_counters(metrics_fair, "counters", fed_dir)

            # metrics_unfair = [
            #     (x, metric) for x, metric in metrics if int(metric["cid"]) >= 75
            # ]
            # (
            #     _,
            #     _,
            #     _,
            #     max_disparity_statistics_unfair,
            #     _,
            # ) = AggregationFunctions.handle_counters(
            #     metrics_unfair, "counters", fed_dir
            # )
            # write avg probabilities to file
            # with open(f"{fed_dir}/avg_proba.pkl", "wb") as file:
            #     dill.dump(average_probabilities, file)
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

        elif args.metric == "error_rate":
            for _, metric in metrics:
                agg_metrics = {}
                agg_metrics["FL Round"] = server_round
                node_name = metric["cid"]
                if "max_error_rate_test" in metric:
                    agg_metrics[f"Test Node {node_name} - Error Rate (Softmax)"] = metric["max_error_rate_test"]
                if "test_accuracy" in metric:
                    agg_metrics[f"Test Node {node_name} - Acc."] = metric["test_accuracy"]
                disparity_dataset = metric.get("max_disparity_dataset", 0)
                agg_metrics[f"Test Node {node_name} - Disp. Dataset"] = disparity_dataset
                if wandb_run:
                    wandb_run.log(agg_metrics)

            (
                _,
                group_accuracy,
                error_rate_group,
                max_error_rate_clients,
                _,
            ) = AggregationFunctions.handle_counters_error_rate_train(metrics, "counters", args)

            for key, value in group_accuracy.items():
                if wandb_run:
                    wandb_run.log(
                        {
                            "FL Round": server_round,
                            f"Test Group Accuracy {key}": value,
                        }
                    )
            for key, value in error_rate_group.items():
                if wandb_run:
                    wandb_run.log(
                        {
                            "FL Round": server_round,
                            f"Test Group Error Rate {key}": value,
                        }
                    )
            (
                error_rate,
                counter_error_rate,
                test_disparity,
            ) = AggregationFunctions.handle_counters_error_rate(metrics, "counters", args)

        if args.metric == "disparity":
            agg_metrics = {
                "Test Loss": loss_test,
                "Test Accuracy": accuracy_test,
                # "Test Disparity with average": max_disparity_average,
                # "Test Disparity with weighted average": max_disparity_weighted_average,
                "Test Disparity with statistics": max_disparity_statistics,
                # "Test Disparity with statistics FAIR": max_disparity_statistics_fair,
                # "Test Disparity with statistics UNFAIR": max_disparity_statistics_unfair,
                "FL Round": server_round,
                # "Test Counter 0|0": sum_counters["0|0"],
                # "Test Counter 0|1": sum_counters["0|1"],
                # "Test Counter 1|0": sum_counters["1|0"],
                # "Test Counter 1|1": sum_counters["1|1"],
                # "Test Target 0": sum_targets["0"],
                # "Test Target 1": sum_targets["1"],
                "Test F1": f1_test,
            }
        elif args.metric == "error_rate":
            agg_metrics = {
                "Test Loss": loss_test,
                "Test Accuracy": accuracy_test,
                "Test Error Rate with statistics": error_rate,
                "FL Round": server_round,
                "Test F1": f1_test,
                "Test Disparity with statistics": test_disparity,
            }
            for key, value in max_error_rate_clients.items():
                if wandb_run:
                    wandb_run.log(
                        {
                            "FL Round": server_round,
                            f"Test Max Error Rate Client {key} - (Argmax)": value,
                        }
                    )

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
                    n_examples * metric["test_loss" if not train_parameters.sweep else "validation_loss"]
                    for n_examples, metric in metrics
                ]
            )
            / total_examples
        )
        accuracy_evaluation = (
            sum(
                [
                    n_examples * metric["test_accuracy" if not train_parameters.sweep else "validation_accuracy"]
                    for n_examples, metric in metrics
                ]
            )
            / total_examples
        )
        f1_validation = sum([n_examples * metric["f1_score"] for n_examples, metric in metrics]) / total_examples
        # max_disparity_average = np.mean(
        #     [
        #         metric[
        #             "max_disparity_test"
        #             if not train_parameters.sweep
        #             else "max_disparity_validation"
        #         ]
        #         for n_examples, metric in metrics
        #     ]
        # )

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

        elif args.metric == "error_rate":
            (
                _,
                group_accuracy,
                error_rate_group,
                max_error_rate_clients,
                _,
            ) = AggregationFunctions.handle_counters_error_rate_train(metrics, "counters", args)

            for key, value in group_accuracy.items():
                if wandb_run:
                    wandb_run.log(
                        {
                            "FL Round": server_round,
                            f"Validation Group Accuracy {key}": value,
                        }
                    )
            for key, value in error_rate_group.items():
                if wandb_run:
                    wandb_run.log(
                        {
                            "FL Round": server_round,
                            f"Validation Group Error Rate {key}": value,
                        }
                    )
            (
                error_rate,
                counter_error_rate,
                max_disparity,
            ) = AggregationFunctions.handle_counters_error_rate(metrics, "counters", args)

        custom_metric = accuracy_evaluation
        if args.target:
            if args.metric == "disparity":
                distance = args.target - max_disparity_statistics
            elif args.metric == "error_rate":
                distance = args.target - error_rate

            if distance > 0:
                penalty = 0
            else:
                penalty = -float("inf")

            custom_metric = accuracy_evaluation + penalty

        if args.metric == "disparity":
            agg_metrics = {
                "Validation Loss": loss_evaluation,
                "Validation_Accuracy": accuracy_evaluation,
                # "Validation Disparity with average": max_disparity_average,
                # "Validation Disparity with weighted average": max_disparity_weighted_average,
                "Validation Disparity with statistics": max_disparity_statistics,
                "Custom_metric": custom_metric,
                "FL Round": server_round,
                # "Validation Counter 0|0": sum_counters["0|0"],
                # "Validation Counter 0|1": sum_counters["0|1"],
                # "Validation Counter 1|0": sum_counters["1|0"],
                # "Validation Counter 1|1": sum_counters["1|1"],
                # "Validation Target 0": sum_targets["0"],
                # "Validation Target 1": sum_targets["1"],
                "Validation F1": f1_validation,
            }

        elif args.metric == "error_rate":
            agg_metrics = {
                "Validation Loss": loss_evaluation,
                "Validation Accuracy": accuracy_evaluation,
                "Validation Error Rate with statistics": error_rate,
                "FL Round": server_round,
                "Custom_metric": custom_metric,
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
        train_parameters,
        unfairness_history,
        wandb_run=None,
        args=None,
    ) -> dict:
        losses = []
        losses_with_regularization = []
        epsilon_list = []
        accuracies = []
        lambda_list = []

        total_examples = sum([n_examples for n_examples, _ in metrics])
        agg_metrics = {
            "FL Round": server_round,
        }
        # wandb_run.log(
        #     {
        #         "FL Round": server_round,
        #         "Alpha": train_parameters.alpha,
        #         "Variance": np.var(unfairness_history),
        #     }
        # )
        # Generic statistics that are logged for each round
        # and that are not dependent on the metric we are using

        all_lambdas = []
        all_disparities = []

        for n_examples, node_metrics in metrics:
            losses.append(n_examples * node_metrics["train_loss"])

            current_target = node_metrics.get("current_target", None)

            losses_with_regularization.append(n_examples * node_metrics["train_loss_with_regularization"])
            epsilon_list.append(node_metrics["epsilon"])
            accuracies.append(n_examples * node_metrics["train_accuracy"])
            lambda_list.append(node_metrics["Lambda"])
            client_id = node_metrics["cid"]
            DPL_lambda = node_metrics["Lambda"]

            if train_parameters.regularization:
                all_lambdas_client = torch.tensor(node_metrics["history_lambda"], device="cpu")
                all_lambdas.append(all_lambdas_client)

            if not args.metric == "error_rate":
                all_disparity_client = torch.tensor(node_metrics["history_disparity"], device="cpu")
                all_disparities.append(all_disparity_client)

            if DPL_lambda:
                agg_metrics[f"Lambda Client {client_id}"] = DPL_lambda

            if args.metric == "disparity":
                disparity_client_after_local_epoch = node_metrics["Disparity Train"]
                agg_metrics[f"Disparity Client {client_id} After Local train"] = float(
                    disparity_client_after_local_epoch.item()
                )

                accuracy_client_after_local_epoch = node_metrics["train_accuracy"]
                agg_metrics[f"Accuracy Client {client_id} After Local train"] = float(accuracy_client_after_local_epoch)

            elif args.metric == "error_rate":
                error_rate_client_after_local_epoch = node_metrics["Error Rate Train"]
                agg_metrics[f"Error Rate Client {client_id} After Local train"] = (error_rate_client_after_local_epoch,)

        min_len = 9999999999

        if train_parameters.regularization:
            for item in all_lambdas:
                min_len = min(min_len, len(list(item)))

            new_all_lambdas = []
            for item in all_lambdas:
                item = list(item)
                item = item[:min_len]
                new_all_lambdas.append(item)

            all_lambdas = np.array(new_all_lambdas)
            all_lambdas_client_mean = np.mean(all_lambdas, axis=0)
            for i, lambda_value in enumerate(all_lambdas_client_mean):
                wandb_run.log(
                    {
                        "time": len(all_lambdas_client_mean) * server_round + i,
                        "History Lambda": lambda_value,
                    }
                )

        if not args.metric == "error_rate":
            for item in all_disparities:
                min_len = min(min_len, len(list(item)))

            new_all_disparities = []
            for item in all_disparities:
                item = list(item)
                item = item[:min_len]
                new_all_disparities.append(item)

            all_disparities = np.array(new_all_disparities)
            all_disparities_mean = np.mean(all_disparities, axis=0)

            for i, disparity_value in enumerate(all_disparities_mean):
                wandb_run.log(
                    {
                        "time": len(all_disparities_mean) * server_round + i,
                        "History Disparity": disparity_value,
                    }
                )

        current_max_epsilon = max(current_max_epsilon, *epsilon_list)
        agg_metrics["Train Loss"] = sum(losses) / total_examples
        agg_metrics["Train Accuracy"] = sum(accuracies) / total_examples
        agg_metrics["Train Loss with Regularization"] = sum(losses_with_regularization) / total_examples
        agg_metrics["Aggregated Lambda"] = (
            sum(lambda_list) / len(lambda_list) if args.regularization_mode == "tunable" else args.regularization_lambda
        )

        agg_metrics["Train Epsilon"] = current_max_epsilon

        # if args.metric == "disparity":
        #     agg_metrics[
        #         "Training Disparity with average" : sum(max_disparity_train)
        #         / len(max_disparity_train)
        #     ]

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

            with open(f"{fed_dir}/sum_counters.pkl", "wb") as file:
                dill.dump(sum_counters, file)
                print("Sum counters saved to disk ", sum_counters)

            (
                sum_counters_no_noise,
                sum_targets_no_noise,
                _,
                max_disparity_statistics_no_noise,
                disparity_combinations_no_noise,
            ) = AggregationFunctions.handle_counters(metrics, "counters_no_noise", fed_dir)
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
            agg_metrics["Training Disparity with statistics no noise"] = max_disparity_statistics_no_noise
            if wandb_run:
                wandb_run.log(
                    {
                        "Training Disparity with statistics": max_disparity_statistics,
                        "Training Disparity with statistics no noise": max_disparity_statistics_no_noise,
                        "FL Round": server_round,
                        # "Training Counter 0|0": sum_counters["0|0"],
                        # "Training Counter 0|1": sum_counters["0|1"],
                        # "Training Counter 1|0": sum_counters["1|0"],
                        # "Training Counter 1|1": sum_counters["1|1"],
                        # "Training Counter 0|0 no noise": sum_counters_no_noise["0|0"],
                        # "Training Counter 0|1 no noise": sum_counters_no_noise["0|1"],
                        # "Training Counter 1|0 no noise": sum_counters_no_noise["1|0"],
                        # "Training Counter 1|1 no noise": sum_counters_no_noise["1|1"],
                        # "Training Target 0": sum_targets["0"],
                        # "Training Target 1": sum_targets["1"],
                        "Average Probabilities": average_probabilities,
                        "current_target": current_target,
                    }
                )
        elif args.metric == "error_rate":
            (
                _,
                group_accuracy,
                error_rate_group,
                max_error_rate_clients,
                sum_counters,
            ) = AggregationFunctions.handle_counters_error_rate_train(metrics, "counters", args, fed_dir)

            with open(f"{fed_dir}/sum_counters.pkl", "wb") as file:
                dill.dump(sum_counters, file)
                print("Sum counters saved to disk ", sum_counters)

            for key, value in group_accuracy.items():
                if wandb_run:
                    wandb_run.log(
                        {
                            "FL Round": server_round,
                            f"Train Group Accuracy {key}": value,
                        }
                    )
            for key, value in error_rate_group.items():
                if wandb_run:
                    wandb_run.log(
                        {
                            "FL Round": server_round,
                            f"Train Group Error Rate {key}": value,
                        }
                    )

            (
                error_rate_no_noise,
                counter_error_rate_no_noise,
                max_disparity,
            ) = AggregationFunctions.handle_counters_error_rate(metrics, "counters_no_noise", args)

            wandb_run.log(
                {
                    "Training Error Rate with statistics NO NOISE": error_rate_no_noise,
                    "FL Round": server_round,
                }
            )
            # for key, value in max_error_rate_clients.items():
            #     if wandb_run:
            #         wandb_run.log(
            #             {
            #                 "FL Round": server_round,
            #                 f"Test Max Error Rate Client {key}": value,
            #             }
            #         )

            average_probabilities = AggregationFunctions.handle_probabilities_error_rate(metrics, "probabilities", args)
            # load the current average_probabilities file
            # and update it with the new values
            current_avg_proba = None
            if os.path.exists(f"{fed_dir}/avg_proba.pkl"):
                with open(f"{fed_dir}/avg_proba.pkl", "rb") as file:
                    current_avg_proba = dill.load(file)
                for key, value in average_probabilities.items():
                    current_avg_proba[key] = value
            if current_avg_proba:
                with open(f"{fed_dir}/avg_proba.pkl", "wb") as file:
                    dill.dump(current_avg_proba, file)
            else:
                with open(f"{fed_dir}/avg_proba.pkl", "wb") as file:
                    dill.dump(average_probabilities, file)

        return agg_metrics

    def handle_counters_error_rate_train(metrics, key_name, args, fed_dir=None):
        group_accuracy = {}
        for _, metric in metrics:
            accuracy_per_group = metric["accuracy_per_group"]
            for group_name, accuracy in accuracy_per_group.items():
                if group_name not in group_accuracy:
                    group_accuracy[group_name] = []
                group_accuracy[group_name].append(accuracy)
        group_accuracy = {k: sum(v) / len(v) for k, v in group_accuracy.items()}

        error_rate_group = {}
        max_error_rate_clients = {}
        possible_sensitive_groups = list(set(args.privileged_group + args.unprivileged_group))
        for _, metric in metrics:
            error_rate_per_group = metric["error_rate_per_group"]
            max_error_rate = 0
            for sens_group in possible_sensitive_groups:
                # inside error_rate_per_group there is a tuple with
                # counter of error and total size of the group
                if sens_group not in error_rate_group:
                    error_rate_group[sens_group] = []
                if sens_group in error_rate_per_group:
                    error_rate_group[sens_group].append(error_rate_per_group[sens_group])

            for unprivileged in list(args.unprivileged_group):
                for privileged in list(args.privileged_group):
                    if unprivileged in error_rate_per_group and privileged in error_rate_per_group:
                        unpr = error_rate_per_group[unprivileged][0] / error_rate_per_group[unprivileged][1]
                        pr = error_rate_per_group[privileged][0] / error_rate_per_group[privileged][1]
                        diff = unpr - pr
                        if diff > max_error_rate:
                            max_error_rate = diff

            max_error_rate_clients[metric["cid"]] = max_error_rate

        for k, values in error_rate_group.items():
            errors = sum([item[0] for item in values])
            total = sum([item[1] for item in values])
            # print("ERRORS", errors)
            # print("TOTAL", total)
            if total > 0:
                error_rate_group[k] = errors / total

        error_rate = 0

        if fed_dir:
            # open the metadata file and update the counters
            with open(f"{fed_dir}/metadata.json", "r") as infile:
                json_file = json.load(infile)

            combinations = json_file["combinations"]  # ["1|0", "1|1"]
            all_combinations = json_file["all_combinations"]  # ["0|0", "0|1", "1|0", "1|1"]
            missing_combinations = json_file["missing_combinations"]  # [("0|0", "1|0"), ("0|1", "1|1")]
            # sum_counters = {"0|0": 0, "0|1": 0, "1|0": 0, "1|1": 0}
            sum_counters = {key: 0 for key in all_combinations}
            possible_sensitive_attributes = json_file["possible_z"]
            possible_targets = json_file["possible_y"]

            sum_possible_sensitive_attributes = {key: 0 for key in possible_sensitive_attributes}  # {"0": 0, "1": 0}

            for _, metric in metrics:
                metric = metric[key_name]
                for combination in combinations:
                    try:
                        # here the combinations that we are considering are only the ones specified
                        # in the metadata json file. If the problem is binary then we will only have two
                        # combinations.
                        sum_counters[combination] += metric[combination]
                    except:
                        continue

                for sensitive_attribute in possible_sensitive_attributes:
                    try:
                        # we count the occurrences of samples with each of the possible sensitive values
                        sum_possible_sensitive_attributes[sensitive_attribute] += metric[sensitive_attribute]
                    except:
                        continue

            # we compute the missing counters from the information that we already have
            for non_existing, existing in missing_combinations:
                sum_counters[non_existing] = (
                    sum_possible_sensitive_attributes[existing[-1]] - sum_counters[existing]
                    if sum_possible_sensitive_attributes[existing[-1]] - sum_counters[existing] > 0
                    else 0
                )
        else:
            sum_counters = None

        return (
            error_rate,
            group_accuracy,  # ok
            error_rate_group,  # ok
            max_error_rate_clients,  # ok
            sum_counters,
        )

    def handle_probabilities_error_rate(metrics, key_name, args):
        average_probabilities = {}

        for _, metric in metrics:
            probabilities = metric[key_name]
            for group in args.privileged_group + args.unprivileged_group:
                if f"{group}_denominator" in probabilities:
                    if f"{group}_denominator" not in average_probabilities:
                        average_probabilities[f"{group}_denominator"] = probabilities[f"{group}_denominator"]
                    else:
                        average_probabilities[f"{group}_denominator"] += probabilities[f"{group}_denominator"]
                if f"{group}_numerator" in probabilities:
                    if f"{group}_numerator" not in average_probabilities:
                        average_probabilities[f"{group}_numerator"] = probabilities[f"{group}_numerator"]
                    else:
                        average_probabilities[f"{group}_numerator"] += probabilities[f"{group}_numerator"]

        final_average_probabilities = {}
        for group in args.privileged_group + args.unprivileged_group:
            if f"{group}_denominator" in average_probabilities and average_probabilities[f"{group}_denominator"] > 0:
                proba = average_probabilities[f"{group}_numerator"] / average_probabilities[f"{group}_denominator"]
                # we want the error rate to be in the range [0,1]
                # because the error rate is always positive. The problem
                # in our case is that since we are using Differential Privacy
                # then we need to be sure that the noise that we introduce
                # does not create a negative error rate or an error rate
                # greater than 1. This is why we need to clip the error rate
                # to be in the range [0,1].
                # final_average_probabilities is in the form
                # {3: 0.44269662921348313, 4: 0.34490238611713664, 2: 0.3389105058365759, 1: 0.08802395209580838, 0: 0.37143772014089016}
                # where the key is the group and the value is the error rate for that group
                if proba > 1:
                    proba = 1
                if proba < 0:
                    proba = 0
                final_average_probabilities[group] = proba

        return final_average_probabilities

    def handle_counters_error_rate(metrics, key_name, args):
        counters = {}

        for group in args.privileged_group + args.unprivileged_group:
            counters[f"{group}_fp"] = 0
            counters[f"{group}_tn"] = 0
            counters[f"{group}_tp"] = 0
            counters[f"{group}_fn"] = 0

        for _, metric in metrics:
            metric = metric[key_name]
            for key in counters.keys():
                try:
                    counters[key] += metric[key]
                except:
                    continue

        errors = []

        for unprivileged in list(args.unprivileged_group):
            den = (
                counters[f"{unprivileged}_fp"]
                + counters[f"{unprivileged}_tn"]
                + counters[f"{unprivileged}_tp"]
                + counters[f"{unprivileged}_fn"]
            )
            if den > 0:
                err_unpriv = (counters[f"{unprivileged}_fp"] + counters[f"{unprivileged}_fn"]) / den
                for privileged in list(args.privileged_group):
                    priv_den = (
                        counters[f"{privileged}_fp"]
                        + counters[f"{privileged}_tn"]
                        + counters[f"{privileged}_tp"]
                        + counters[f"{privileged}_fn"]
                    )
                    if priv_den > 0:
                        err_priv = (counters[f"{privileged}_fp"] + counters[f"{privileged}_fn"]) / priv_den
                        error_rate = err_unpriv - err_priv if err_unpriv - err_priv > 0 else 0
                        errors.append(error_rate)
                # error_rate = abs(err_unpriv - err_priv) # if err_unpriv - err_priv > 0 else 0
        # err_unpriv = (counters["fp_unprivileged"] + counters["fn_unprivileged"]) / (counters["fp_unprivileged"] + counters["tn_unprivileged"] + counters["tp_unprivileged"] + counters["fn_unprivileged"])
        # err_priv = (counters["fp_privileged"] + counters["fn_privileged"]) / (counters["fp_privileged"] + counters["tn_privileged"] + counters["tp_privileged"] + counters["fn_privileged"])
        # error_rate = err_unpriv - err_priv if err_unpriv - err_priv > 0 else 0
        # error_rate = abs(err_unpriv - err_priv) # if err_unpriv - err_priv > 0 else 0

        # Beside the error rate, I also want to compute the
        # disparity between the different groups

        disp_counters = {}
        for _, metric in metrics:
            metric = metric[key_name]
            for key in metric.keys():
                if key in disp_counters:
                    disp_counters[key] += metric[key]
                else:
                    disp_counters[key] = metric[key]

        print(disp_counters)
        combinations = []
        for unprivileged in list(args.unprivileged_group):
            for privileged in list(args.privileged_group):
                if unprivileged != privileged and (unprivileged, privileged) not in combinations:
                    combinations.append((unprivileged, privileged))

        possible_targets = []
        for key in disp_counters.keys():
            if "|" in key:
                possible_targets.append(key.split("|")[0])

        max_disparity = 0
        for target in possible_targets:
            for unprivileged, privileged in combinations:
                unpriv_den = disp_counters[f"{unprivileged}"]
                unpriv_num = disp_counters[f"{target}|{unprivileged}"]

                priv_den = disp_counters[f"{privileged}"]
                priv_num = disp_counters[f"{target}|{privileged}"]

                max_disparity = max(
                    max_disparity,
                    abs(unpriv_num / unpriv_den - priv_num / priv_den),
                )

        return max(errors), counters, max_disparity

    def handle_counters(metrics, key, fed_dir):
        # open the metadata file and update the counters
        with open(f"{fed_dir}/metadata.json", "r") as infile:
            json_file = json.load(infile)

        combinations = json_file["combinations"]  # ["1|0", "1|1"]
        all_combinations = json_file["all_combinations"]  # ["0|0", "0|1", "1|0", "1|1"]
        missing_combinations = json_file["missing_combinations"]  # [("0|0", "1|0"), ("0|1", "1|1")]
        # sum_counters = {"0|0": 0, "0|1": 0, "1|0": 0, "1|1": 0}
        sum_counters = {key: 0 for key in all_combinations}
        possible_sensitive_attributes = json_file["possible_z"]
        possible_targets = json_file["possible_y"]

        sum_possible_sensitive_attributes = {key: 0 for key in possible_sensitive_attributes}  # {"0": 0, "1": 0}

        for _, metric in metrics:
            metric = metric[key]
            for combination in combinations:
                try:
                    # here the combinations that we are considering are only the ones specified
                    # in the metadata json file. If the problem is binary then we will only have two
                    # combinations.
                    sum_counters[combination] += metric[combination]
                except:
                    continue

            for sensitive_attribute in possible_sensitive_attributes:
                try:
                    # we count the occurrences of samples with each of the possible sensitive values
                    sum_possible_sensitive_attributes[sensitive_attribute] += metric[sensitive_attribute]
                except:
                    continue

        # we compute the missing counters from the information that we already have
        for non_existing, existing in missing_combinations:
            sum_counters[non_existing] = (
                sum_possible_sensitive_attributes[existing[-1]] - sum_counters[existing]
                if sum_possible_sensitive_attributes[existing[-1]] - sum_counters[existing] > 0
                else 0
            )
        average_probabilities = {}
        for combination in all_combinations:
            try:
                proba = sum_counters[combination] / sum_possible_sensitive_attributes[combination[2]]
                if proba > 1:
                    proba = 1
                if proba < 0:
                    proba = 0
                average_probabilities[combination] = proba
            except:
                print("Error in computing the average probabilities")
                continue

        max_disparity_statistics = []
        combinations_disparity = []

        print(
            "POSSIBLE TARGETS",
            possible_targets,
            "possible sensitive",
            possible_sensitive_attributes,
            "SUm counters: ",
            sum_counters,
            "Average proba:",
            average_probabilities,
        )
        for target in possible_targets:  # ["0"]:
            for sensitive_value in possible_sensitive_attributes:
                Y_target_Z_sensitive_value = sum_counters[f"{target}|{sensitive_value}"]
                Z_sensitive_value = sum_possible_sensitive_attributes[sensitive_value]
                Z_not_sensitive_value = 0
                Y_target_Z_not_sensitive_value = 0
                for not_sensitive_value in possible_sensitive_attributes:
                    if not_sensitive_value != sensitive_value:
                        Y_target_Z_not_sensitive_value += sum_counters[f"{target}|{not_sensitive_value}"]
                        Z_not_sensitive_value += sum_possible_sensitive_attributes[not_sensitive_value]

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
            (target, sv, disparity) for (target, sv), disparity in zip(combinations_disparity, max_disparity_statistics)
        ]

        return (
            sum_counters,
            sum_possible_sensitive_attributes,
            average_probabilities,
            max_disparity_with_statistics,  # max_disparity_statistics,
            combinations,
        )
