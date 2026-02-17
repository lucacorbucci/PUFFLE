# ABOUTME: MMD-Fair FedAvg competitor method package.
# ABOUTME: Implements fairness via function tracking with MMD kernel.

from competitors.mmd_fair.model import MMDFairModel, distance_kernel

__all__ = ["MMDFairModel", "distance_kernel"]
