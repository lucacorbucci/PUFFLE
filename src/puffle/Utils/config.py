from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from puffle.Utils.constants import (
    DEFAULT_ALPHA,
    DEFAULT_MOMENTUM,
    DEFAULT_WEIGHT_DECAY_ALPHA,
)
from puffle.Utils.lambda_updater import LambdaUpdateStrategy


class FairnessMetric(str, Enum):
    """Supported fairness metrics for PUFFLE."""

    DISPARITY = "disparity"
    ERROR_RATE = "error_rate"


class PUFFLEConfig(BaseModel):
    """Configuration for PUFFLE model hyperparameters and behavior."""

    model_config = ConfigDict(frozen=True, use_enum_values=False)

    lambda_regularization: float = Field(default=0.0, ge=0.0, le=1.0)
    target: float | None = Field(default=None, ge=0.0, le=1.0)
    momentum: float = Field(default=DEFAULT_MOMENTUM, ge=0.0, le=1.0)
    alpha: float = Field(default=DEFAULT_ALPHA, gt=0.0)
    weight_decay_alpha: float = Field(
        default=DEFAULT_WEIGHT_DECAY_ALPHA, ge=0.0, le=1.0
    )
    tunable_lambda: bool = False
    lambda_update_strategy: LambdaUpdateStrategy = LambdaUpdateStrategy.GRADIENT
    fairness_metric: FairnessMetric = FairnessMetric.DISPARITY

    # PID-specific parameters
    lambda_kp: float = Field(default=0.01, ge=0.0)
    lambda_ki: float = Field(default=0.001, ge=0.0)
    lambda_kd: float = Field(default=0.005, ge=0.0)

    sigma_update_lambda: float | None = Field(default=None, ge=0.0)
    sigma_statistics: float | None = Field(default=None, ge=0.0)
