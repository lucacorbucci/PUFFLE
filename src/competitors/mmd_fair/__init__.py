# ABOUTME: MMD-Fair FedAvg competitor method package.
# ABOUTME: Implements fairness via function tracking with MMD kernel.

from competitors.mmd_fair.model import MMDFairModel, distance_kernel
from competitors.mmd_fair.prediction_tracker import PredictionTracker

__all__ = ["MMDFairModel", "PredictionTracker", "distance_kernel"]
