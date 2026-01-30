from unittest.mock import MagicMock

import pytest

from FlowerFLTemplate.Aggregations.aggregations import Aggregation
from puffle.Utils.modes import MetricMode


class TestDatasetDisparityAggregation:
    def test_dataset_disparity_logic(self):
        """
        Verify that Aggregation correctly aggregates dataset counters 
        and computes Dataset Disparity.
        """
        # Mock metrics from 2 clients
        # Client A: Z=0 only, Y=1 (Bias in data: 100% positive for Z=0)
        metrics_a = {
            "dataset_counter_z": 0,
            "dataset_counter_not_z": 10,
            "dataset_counter_y_z": 0,
            "dataset_counter_y_not_z": 10,
            "client_id": 1
        }
        
        # Client B: Z=1 only, Y=0 (Bias in data: 0% positive for Z=1)
        metrics_b = {
            "dataset_counter_z": 10,
            "dataset_counter_not_z": 0,
            "dataset_counter_y_z": 0,
            "dataset_counter_y_not_z": 0,
            "client_id": 2
        }
        
        # Tuple format expected by Aggregation: (num_examples, metric_dict)
        metrics = [(10, metrics_a), (10, metrics_b)]
        
        server_round = 1
        wandb_run = MagicMock()
        
        # Call Aggregation
        agg_results = Aggregation.agg_metrics_evaluation(
            metrics=metrics, 
            server_round=server_round, 
            wandb_run=wandb_run
        )
        
        # Check Results
        # Total Z=1: 10 (from B)
        # Total Z=0: 10 (from A)
        # Total Y=1|Z=1: 0 (from B) -> P(Y=1|Z=1) = 0.0
        # Total Y=1|Z=0: 10 (from A) -> P(Y=1|Z=0) = 1.0
        # Expected Disparity = |1.0 - 0.0| = 1.0
        
        print(f"Aggregated Results: {agg_results}")
        
        assert "Validation Dataset Disparity" in agg_results
        disparity = agg_results["Validation Dataset Disparity"]
        assert disparity == 1.0, f"Expected 1.0, got {disparity}"
        
        # Verify WandB logging calls
        # We expect a log call with 'Validation_Dataset_Disparity'
        wandb_calls = [c[0][0] for c in wandb_run.log.call_args_list]
        keys_logged = []
        for call_arg in wandb_calls:
            keys_logged.extend(call_arg.keys())
            
        assert "Validation_Dataset_Disparity" in keys_logged
