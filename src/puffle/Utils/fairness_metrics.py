# ABOUTME: Structured representation of fairness evaluation results.
# ABOUTME: Provides dictionary-like access for backward compatibility.

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FairnessMetrics:
    """Standard set of metrics for fairness-aware model evaluation."""

    loss: float
    accuracy: float
    f1: float
    disparity: float
    statistics: dict = field(default_factory=dict)
    dataset_disparity: float = 0.0
    dataset_statistics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert metrics to a flat dictionary."""
        return {
            "loss": float(self.loss),
            "accuracy": float(self.accuracy),
            "f1": float(self.f1),
            "disparity": float(self.disparity),
            "statistics": self.statistics,
            "dataset_disparity": float(self.dataset_disparity),
            "dataset_statistics": self.dataset_statistics,
        }

    def __getitem__(self, key: str):
        """Allow dict-like access for backward compatibility."""
        if not isinstance(key, str):
            msg = f"Key must be a string, got {type(key)}"
            raise TypeError(msg)
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __contains__(self, key: str) -> bool:
        """Allow 'in' operator for backward compatibility."""
        return hasattr(self, key)

    def get(self, key: str, default=None):
        """Allow .get() for backward compatibility."""
        try:
            return self[key]
        except KeyError:
            return default
