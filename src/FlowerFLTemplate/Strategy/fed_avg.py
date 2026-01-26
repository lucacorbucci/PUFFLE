# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Federated Averaging (FedAvg) [McMahan et al., 2016] strategy.

Paper: arxiv.org/abs/1602.05629
"""

from collections.abc import Callable
from logging import WARNING
from typing import Any

from ClientManager.client_manager import SimpleClientManager
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace, weighted_loss_avg
from flwr.server.strategy.strategy import Strategy
from Utils.preferences import Preferences

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


# pylint: disable=line-too-long
class FedAvg(Strategy):
    """
    Federated Averaging strategy implementation.

    Based on McMahan et al. (2016). Supports client sampling for fit/evaluate/test, weighted aggregation of parameters and metrics, initial parameters, and custom config functions.

    Args:
        fraction_fit (float, optional): Fraction of clients for training. Defaults to 1.0.
        fraction_evaluate (float, optional): Fraction of clients for evaluation. Defaults to 1.0.
        min_fit_clients (int, optional): Minimum clients for training. Defaults to 2.
        min_evaluate_clients (int, optional): Minimum clients for evaluation. Defaults to 2.
        min_available_clients (int, optional): Minimum total clients. Defaults to 2.
        preferences (Preferences): User preferences containing FL settings.
        evaluate_fn (Callable, optional): Central evaluation function. Defaults to None.
        on_fit_config_fn (Callable, optional): Config for training round. Defaults to None.
        on_evaluate_config_fn (Callable, optional): Config for evaluation round. Defaults to None.
        accept_failures (bool, optional): Accept rounds with failures. Defaults to True.
        initial_parameters (Parameters, optional): Initial global parameters. Defaults to None.
        fit_metrics_aggregation_fn (MetricsAggregationFn, optional): Aggregation for fit metrics. Defaults to None.
        evaluate_metrics_aggregation_fn (MetricsAggregationFn, optional): Aggregation for evaluate metrics. Defaults to None.
        test_metrics_aggregation_fn (MetricsAggregationFn, optional): Aggregation for test metrics. Defaults to None.
        inplace (bool, optional): In-place aggregation. Defaults to True.
        wandb_run (Any, optional): WandB run for logging. Defaults to None.

    Returns:
        None
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        preferences: Preferences,
        evaluate_fn: Callable[[int, NDArrays, dict[str, Scalar]], tuple[float, dict[str, Scalar]] | None] | None = None,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        initial_parameters: Parameters | None = None,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        test_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        inplace: bool = True,
        wandb_run: Any = None,
    ) -> None:
        """
        Initializes the FedAvg strategy with sampling and aggregation parameters.

        Validates min_available_clients; sets attributes for client sampling, config functions, aggregation, etc.

        Args:
            fraction_fit (float, optional): Fraction of clients for fit. Defaults to 1.0.
            fraction_evaluate (float, optional): Fraction of clients for evaluate. Defaults to 1.0.
            min_fit_clients (int, optional): Min clients for fit. Defaults to 2.
            min_evaluate_clients (int, optional): Min clients for evaluate. Defaults to 2.
            min_available_clients (int, optional): Min total clients. Defaults to 2.
            preferences (Preferences): FL preferences.
            evaluate_fn (Callable, optional): Central eval function. Defaults to None.
            on_fit_config_fn (Callable, optional): Fit config fn. Defaults to None.
            on_evaluate_config_fn (Callable, optional): Evaluate config fn. Defaults to None.
            accept_failures (bool, optional): Accept failures. Defaults to True.
            initial_parameters (Parameters, optional): Initial params. Defaults to None.
            fit_metrics_aggregation_fn (MetricsAggregationFn, optional): Fit metrics agg fn. Defaults to None.
            evaluate_metrics_aggregation_fn (MetricsAggregationFn, optional): Evaluate metrics agg fn. Defaults to None.
            test_metrics_aggregation_fn (MetricsAggregationFn, optional): Test metrics agg fn. Defaults to None.
            inplace (bool, optional): In-place agg. Defaults to True.
            wandb_run (Any, optional): WandB run. Defaults to None.

        Returns:
            None

        Raises:
            ValueError: If min_available_clients too low (logged as warning).
        """
        super().__init__()

        if min_fit_clients > min_available_clients or min_evaluate_clients > min_available_clients:
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.test_metrics_aggregation_fn = test_metrics_aggregation_fn
        self.wandb_run = wandb_run
        self.fed_dir = preferences.fed_dir

    def __repr__(self) -> str:
        """
        Returns a string representation of the FedAvg strategy.

        Args:
            None

        Returns:
            str: Representation including accept_failures flag.
        """
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """
        Computes the number of clients to sample for fit and the min available required.

        Uses fraction_fit, ensures at least min_fit_clients.

        Args:
            num_available_clients (int): Total available clients.

        Returns:
            tuple[int, int]: (sample_size, min_available_clients)
        """
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """
        Computes the number of clients to sample for evaluation and the min available required.

        Uses fraction_evaluate, ensures at least min_evaluate_clients.

        Args:
            num_available_clients (int): Total available clients.

        Returns:
            tuple[int, int]: (sample_size, min_available_clients)
        """
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(self, client_manager: SimpleClientManager) -> Parameters | None:
        """
        Initializes global model parameters using provided initial_parameters or strategy default.

        Clears initial_parameters from memory after use.

        Args:
            client_manager (SimpleClientManager): Client manager (unused in this implementation).

        Returns:
            Parameters | None: Initial global parameters.
        """
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(self, server_round: int, parameters: Parameters) -> tuple[float, dict[str, Scalar]] | None:
        """
        Performs central evaluation using the provided evaluate_fn.

        Converts parameters to NDArrays, calls evaluate_fn if available.

        Args:
            server_round (int): Current server round.
            parameters (Parameters): Global model parameters.

        Returns:
            tuple[float, dict[str, Scalar]] | None: (loss, metrics) or None if no evaluate_fn.
        """
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: SimpleClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """
        Configures the fit instructions for the next training round.

        Generates config if on_fit_config_fn provided, samples clients using num_fit_clients, creates FitIns.

        Args:
            server_round (int): Current server round.
            parameters (Parameters): Global parameters to send to clients.
            client_manager (SimpleClientManager): Manager to sample clients from.

        Returns:
            list[tuple[ClientProxy, FitIns]]: List of (client, fit instructions) pairs.

        Raises:
            ValueError: If no clients can be sampled.
        """
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available(phase="training"))
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            phase="training",
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: SimpleClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """
        Configures the evaluate instructions for the next validation round.

        Skips if fraction_evaluate is 0; generates config if on_evaluate_config_fn provided, samples clients using num_evaluation_clients for validation phase.

        Args:
            server_round (int): Current server round.
            parameters (Parameters): Global parameters to send to clients.
            client_manager (SimpleClientManager): Manager to sample clients from.

        Returns:
            list[tuple[ClientProxy, EvaluateIns]]: List of (client, evaluate instructions) pairs.

        Raises:
            ValueError: If no clients can be sampled.
        """
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available(phase="validation"))
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            phase="validation",
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def configure_test(
        self, server_round: int, parameters: Parameters, client_manager: SimpleClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """
        Configures the test instructions for the next test round.

        Skips if fraction_evaluate is 0; generates config if on_evaluate_config_fn provided, samples clients using num_evaluation_clients for test phase.

        Args:
            server_round (int): Current server round.
            parameters (Parameters): Global parameters to send to clients.
            client_manager (SimpleClientManager): Manager to sample clients from.

        Returns:
            list[tuple[ClientProxy, EvaluateIns]]: List of (client, test instructions) pairs.

        Raises:
            ValueError: If no clients can be sampled.
        """
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(client_manager.num_available(phase="test"))
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            phase="test",
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """
        Aggregates fit results: weighted average of parameters, custom metrics if fn provided.

        Uses aggregate_inplace if inplace=True; skips if no results or unacceptable failures.

        Args:
            server_round (int): Current server round.
            results (list[tuple[ClientProxy, FitRes]]): Successful fit results.
            failures (list[tuple[ClientProxy, FitRes] | BaseException]): Failures.

        Returns:
            tuple[Parameters | None, dict[str, Scalar]]: Aggregated parameters and metrics.

        Raises:
            ValueError: If aggregation fails.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            aggregated_ndarrays = aggregate_inplace(results)
        else:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results
            ]
            aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(
                metrics=fit_metrics,
                server_round=server_round,
                wandb_run=self.wandb_run,
                fed_dir=self.fed_dir,
            )
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """
        Aggregates evaluate results: weighted average loss, custom metrics if fn provided.

        Skips if no results or unacceptable failures.

        Args:
            server_round (int): Current server round.
            results (list[tuple[ClientProxy, EvaluateRes]]): Successful evaluate results.
            failures (list[tuple[ClientProxy, EvaluateRes] | BaseException]): Failures.

        Returns:
            tuple[float | None, dict[str, Scalar]]: Aggregated loss and metrics.

        Raises:
            ValueError: If aggregation fails.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(
                metrics=eval_metrics, server_round=server_round, wandb_run=self.wandb_run
            )
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated

    def aggregate_test(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """
        Aggregates test results: weighted average loss, custom metrics if fn provided.

        Skips if no results or unacceptable failures.

        Args:
            server_round (int): Current server round.
            results (list[tuple[ClientProxy, EvaluateRes]]): Successful test results.
            failures (list[tuple[ClientProxy, EvaluateRes] | BaseException]): Failures.

        Returns:
            tuple[float | None, dict[str, Scalar]]: Aggregated loss and metrics.

        Raises:
            ValueError: If aggregation fails.
        """
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [(evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results],
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.test_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.test_metrics_aggregation_fn(
                metrics=eval_metrics,
                server_round=server_round,
                wandb_run=self.wandb_run,
            )
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No test_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
