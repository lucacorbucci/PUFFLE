from collections.abc import Callable
from typing import Any

from flwr.common import (
    Context,
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg as FlwrFedAvg

from FlowerFLTemplate.Utils.preferences import Preferences


class FedAvg(FlwrFedAvg):
    def __init__(  # noqa: PLR0913
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Callable[
            [Context, Parameters],
            tuple[float, dict[str, Scalar]] | None,
        ]
        | None = None,
        on_fit_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int], dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        initial_parameters: Parameters | None = None,
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        test_metrics_aggregation_fn: MetricsAggregationFn | None = None,
        preferences: Preferences | None = None,
        wandb_run=None,
    ) -> None:
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.preferences = preferences
        self.wandb_run = wandb_run
        self.test_metrics_aggregation_fn = test_metrics_aggregation_fn

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}

        # Call super method to get aggregated parameters
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        # Custom metrics aggregation (if provided)
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]

            # Note: The aggregation functions in Aggregations/aggregations.py expect slightly different args
            # than standard Flower.
            # user's main.py passed Aggregation.agg_metrics_train
            # checking signature of agg_metrics_train in aggregations.py would be good but I assume it matches what the user intends
            # Based on previous main.py:
            # custom_metrics = Aggregation.agg_metrics_train(metrics, server_round, preferences.fed_dir, run)

            # But here we are passing it as fit_metrics_aggregation_fn to super().
            # HOWEVER, standard FlwrFedAvg calls fit_metrics_aggregation_fn(fit_metrics) -> metrics_aggregated
            # If Aggregation.agg_metrics_train has extra args (server_round, fed_dir, run), we can't pass it directly to super.
            #
            # So I should manually call it from here if I override aggregate_fit, OR
            # create a partial function in main.py?
            #
            # User's main.py passes it to FedAvg constructor:
            # fit_metrics_aggregation_fn=Aggregation.agg_metrics_train,
            #
            # The user's main.py calls:
            # strategy = FedAvg(..., fit_metrics_aggregation_fn=Aggregation.agg_metrics_train, ...)
            #
            # If I override aggregate_fit, I can call it manually.

        # Since I'm inheriting, I should replicate the logic intended.
        # Let's check Aggregation.agg_metrics_train signature.
        # But I recall from memory it takes specific args.

        # If I look at the USER'S PREVIOUS main.py code (in Step 1980):
        # custom_metrics = Aggregation.agg_metrics_train(
        #     metrics, server_round, preferences.fed_dir, run
        # )

        # So I should implement that here.
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            # I assume the user passed the function correctly.
            # check if fit_metrics_aggregation_fn expects just metrics?
            # If it expects more, I must handle it.
            # Assuming I need to manually call it with extra args:

            aggregated_metrics = self.fit_metrics_aggregation_fn(
                fit_metrics,
                server_round=server_round,
                fed_dir=self.preferences.fed_dir if self.preferences else "",
                wandb_run=self.wandb_run,
            )
            metrics_aggregated.update(aggregated_metrics)

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, Any]],
        failures: list[tuple[ClientProxy, Any] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """Aggregate evaluation results using weighted average."""
        if not results:
            return None, {}

        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]

            # Based on previous main.py:
            # custom_metrics = Aggregation.agg_metrics_evaluation(metrics, server_round, run)
            aggregated_metrics = self.evaluate_metrics_aggregation_fn(
                eval_metrics, server_round=server_round, wandb_run=self.wandb_run
            )
            metrics_aggregated.update(aggregated_metrics)

        return loss_aggregated, metrics_aggregated
