# ABOUTME: Flower strategy for MMD-Fair FL simulation.
# ABOUTME: Extends FedAvg with server-side Y_0/Y_1 tracking and alpha weight computation.

import io
from logging import INFO

import numpy as np
import torch
from FlowerFLTemplate.Strategy.fed_avg import FedAvg
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy

from competitors.mmd_fair.prediction_tracker import PredictionTracker


class MMDFairFedAvg(FedAvg):
    """
    FedAvg strategy with MMD-Fair server-side tracking.

    Extends the standard FedAvg strategy with:
    - Server-side Y_0/Y_1 prediction tracking
    - Alpha weight computation per client
    - Prediction sampling and update protocol
    """

    def __init__(
        self,
        *,
        mu: float = 1.0,
        ny: int = 100,
        lambda_fairness: float = 1.0,
        **kwargs,
    ):
        """
        Initialize MMD-Fair FedAvg strategy.

        Args:
            mu: Drop rate for prediction tracking (0.0 to 1.0)
            ny: Capacity of prediction trackers
            lambda_fairness: Fairness regularization weight
            **kwargs: Arguments passed to parent FedAvg

        """
        super().__init__(**kwargs)

        self.mu = mu
        self.ny = ny
        self.lambda_fairness = lambda_fairness

        # Prediction trackers (initialized in initialize_parameters)
        self.Y_0: PredictionTracker | None = None
        self.Y_1: PredictionTracker | None = None

        # Client weights and alpha values (computed after first round)
        self.client_weights: dict[int, float] = {}
        self.alpha_weights: dict[int, tuple[float, float]] = {}
        self.total_samples: int = 0

    def initialize_parameters(self, client_manager) -> Parameters | None:
        """
        Initialize global parameters and prediction trackers.

        Args:
            client_manager: Client manager

        Returns:
            Initial global parameters

        """
        # Initialize parent (gets initial model parameters)
        params = super().initialize_parameters(client_manager)

        # Initialize prediction trackers
        self.Y_0 = PredictionTracker(demographic_group=0, capacity=self.ny)
        self.Y_1 = PredictionTracker(demographic_group=1, capacity=self.ny)

        log(INFO, f"Initialized MMD-Fair trackers: mu={self.mu}, ny={self.ny}")

        return params

    def configure_fit(self, server_round, parameters, client_manager):
        """
        Configure fit instructions with Y_0/Y_1 tracking sets.

        Args:
            server_round: Current round number
            parameters: Global parameters
            client_manager: Client manager

        Returns:
            List of (client, fit_ins) pairs

        """
        # Get base config from parent
        client_instructions = super().configure_fit(
            server_round, parameters, client_manager
        )

        # Serialize Y_0/Y_1 into config
        if self.Y_0 is not None and self.Y_1 is not None:
            Y_0_buffer = io.BytesIO()
            Y_1_buffer = io.BytesIO()
            torch.save(self.Y_0.get_predictions(), Y_0_buffer)
            torch.save(self.Y_1.get_predictions(), Y_1_buffer)

            # Add tracking sets to each client's config
            updated_instructions = []
            for client, fit_ins in client_instructions:
                # Get client ID from properties
                client_id = int(client.cid)

                # Get alpha weights for this client (default to 1.0 if not computed yet)
                alpha_0, alpha_1 = self.alpha_weights.get(client_id, (1.0, 1.0))

                # Update config
                config = dict(fit_ins.config)
                config["Y_0_bytes"] = Y_0_buffer.getvalue()
                config["Y_1_bytes"] = Y_1_buffer.getvalue()
                config["alpha_0"] = alpha_0
                config["alpha_1"] = alpha_1
                config["N"] = self.total_samples if self.total_samples > 0 else 1

                # Create new FitIns with updated config
                from flwr.common import FitIns

                updated_fit_ins = FitIns(parameters=fit_ins.parameters, config=config)
                updated_instructions.append((client, updated_fit_ins))

            return updated_instructions

        return client_instructions

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """
        Aggregate fit results and update prediction trackers.

        Args:
            server_round: Current round
            results: Successful fit results
            failures: Failed results

        Returns:
            Aggregated parameters and metrics

        """
        # Standard FedAvg aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Update client weights and alpha values (first round)
        if server_round == 1:
            self._compute_client_statistics(results)

        # Update prediction trackers (Algorithm 2)
        self._update_predictions(results)

        return aggregated_parameters, aggregated_metrics

    def _compute_client_statistics(
        self, results: list[tuple[ClientProxy, FitRes]]
    ) -> None:
        """
        Compute client weights and alpha values.

        Args:
            results: Fit results from clients

        """
        # Compute client weights (proportional to dataset size)
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        self.total_samples = total_examples

        for client, fit_res in results:
            client_id = int(client.cid)
            self.client_weights[client_id] = fit_res.num_examples / total_examples

        # Compute alpha weights: alpha_k_a = P_k(A=a) / P(A=a)
        # For now, use uniform alpha (would need client demographic info)
        # In full implementation, clients would report P_k(A=0) and P_k(A=1)
        # and we'd compute global P(A=0) and P(A=1) to calculate alpha weights
        for client_id in self.client_weights:
            alpha_0 = 1.0  # P_k(A=0) / P(A=0)
            alpha_1 = 1.0  # P_k(A=1) / P(A=1)
            self.alpha_weights[client_id] = (alpha_0, alpha_1)

        log(INFO, f"Computed weights for {len(self.client_weights)} clients")

    def _update_predictions(self, results: list[tuple[ClientProxy, FitRes]]) -> None:
        """
        Update Y_0/Y_1 trackers with new predictions from clients.

        Implements Algorithm 2 from Fair-FL.

        Args:
            results: Fit results containing prediction samples

        """
        if self.Y_0 is None or self.Y_1 is None:
            return

        # Drop old predictions
        self.Y_0.drop(self.mu)
        self.Y_1.drop(self.mu)

        # Collect new predictions from clients
        new_preds_0 = []
        new_preds_1 = []

        for client, fit_res in results:
            client_id = int(client.cid)
            metrics = fit_res.metrics

            # Deserialize prediction samples
            if "pred_0_bytes" in metrics and "pred_1_bytes" in metrics:
                pred_0_buffer = io.BytesIO(metrics["pred_0_bytes"])  # type: ignore
                pred_1_buffer = io.BytesIO(metrics["pred_1_bytes"])  # type: ignore

                pred_0 = torch.load(pred_0_buffer, weights_only=False)
                pred_1 = torch.load(pred_1_buffer, weights_only=False)

                # Weight by client's alpha and sample proportionally
                weight = self.client_weights.get(client_id, 1.0)
                alpha_0, alpha_1 = self.alpha_weights.get(client_id, (1.0, 1.0))

                # Sample predictions proportional to alpha * weight * mu * ny
                n_sample_0 = int(alpha_0 * weight * self.mu * self.ny)
                n_sample_1 = int(alpha_1 * weight * self.mu * self.ny)

                if len(pred_0) > 0 and n_sample_0 > 0:
                    sample_size_0 = min(n_sample_0, len(pred_0))
                    indices_0 = np.random.choice(
                        len(pred_0), size=sample_size_0, replace=False
                    )
                    new_preds_0.append(pred_0[indices_0])

                if len(pred_1) > 0 and n_sample_1 > 0:
                    sample_size_1 = min(n_sample_1, len(pred_1))
                    indices_1 = np.random.choice(
                        len(pred_1), size=sample_size_1, replace=False
                    )
                    new_preds_1.append(pred_1[indices_1])

        # Update trackers
        if new_preds_0:
            self.Y_0.update(new_preds_0)
        if new_preds_1:
            self.Y_1.update(new_preds_1)

        log(INFO, f"Updated trackers: Y_0={len(self.Y_0)}, Y_1={len(self.Y_1)}")
