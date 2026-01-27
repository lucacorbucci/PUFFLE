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

    def to_dict(self) -> dict:
        """Convert metrics to a flat dictionary."""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "disparity": self.disparity,
            "statistics": self.statistics,
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
