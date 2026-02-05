from enum import Enum


class MetricMode(str, Enum):
    """Mode for metric collection and logging."""

    TRAIN = "train"
    VALIDATION = "val"
    TEST = "test"

    def __str__(self) -> str:
        """Return the string value of the enum."""
        return str(self.value)
