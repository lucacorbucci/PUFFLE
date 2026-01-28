from enum import Enum


class LambdaUpdateStrategy(str, Enum):
    """Strategy for updating lambda during training."""

    MOMENTUM = "momentum"
    GRADIENT = "gradient"
    PID = "pid"


class LambdaUpdater:
    """Handles lambda updates with different strategies."""

    def __init__(
        self,
        strategy: LambdaUpdateStrategy = LambdaUpdateStrategy.GRADIENT,
        alpha: float = 0.01,
        momentum: float = 0.9,
        # PID-specific parameters
        kp: float = 0.01,  # Proportional gain
        ki: float = 0.001,  # Integral gain
        kd: float = 0.005,  # Derivative gain
    ):
        """
        Initialize lambda updater.

        Args:
            strategy: Update strategy to use
            alpha: Learning rate (for gradient and momentum)
            momentum: Momentum coefficient (for momentum strategy)
            kp: Proportional gain (for PID)
            ki: Integral gain (for PID)
            kd: Derivative gain (for PID)
        """
        self.strategy = strategy
        self.alpha = alpha
        self.momentum = momentum

        # PID parameters
        self.kp = kp
        self.ki = ki
        self.kd = kd

        # State variables
        self.velocity = 0.0  # For momentum
        self.integral = 0.0  # For PID
        self.prev_error = 0.0  # For PID

    def update(
        self,
        current_lambda: float,
        unfairness: float,
        target: float,
    ) -> float:
        """
        Update lambda based on the selected strategy.

        Args:
            current_lambda: Current lambda value
            unfairness: Current unfairness metric
            target: Target unfairness value

        Returns:
            Updated lambda value (constrained to [0, 1])

        """
        if self.strategy == LambdaUpdateStrategy.MOMENTUM:
            new_lambda = self._update_momentum(current_lambda, unfairness, target)
        elif self.strategy == LambdaUpdateStrategy.GRADIENT:
            new_lambda = self._update_gradient(current_lambda, unfairness, target)
        elif self.strategy == LambdaUpdateStrategy.PID:
            new_lambda = self._update_pid(current_lambda, unfairness, target)
        else:
            msg = f"Unknown strategy: {self.strategy}"
            raise ValueError(msg)

        # Constrain to [0, 1]
        return max(0.0, min(1.0, new_lambda))

    def _update_momentum(
        self,
        current_lambda: float,
        unfairness: float,
        target: float,
    ) -> float:
        """
        Momentum-based update (original algorithm).

        Uses velocity accumulation to smooth updates but can overshoot.
        """
        delta = target - unfairness
        self.velocity = self.momentum * self.velocity + delta
        return current_lambda - self.velocity * self.alpha

    def _update_gradient(
        self,
        current_lambda: float,
        unfairness: float,
        target: float,
    ) -> float:
        """
        Direct gradient-based update (current algorithm).

        Simple proportional response to error.
        """
        return current_lambda + self.alpha * (unfairness - target)

    def _update_pid(
        self,
        current_lambda: float,
        unfairness: float,
        target: float,
    ) -> float:
        """
        PID controller update (new algorithm).

        Combines:
        - Proportional: Responds to current error
        - Integral: Eliminates steady-state error
        - Derivative: Dampens oscillations

        This is the recommended approach for balancing responsiveness
        and stability.
        """
        error = unfairness - target

        # Proportional term: responds to current error
        p_term = self.kp * error

        # Integral term: accumulates error over time
        self.integral += error
        i_term = self.ki * self.integral

        # Derivative term: responds to rate of change
        derivative = error - self.prev_error
        d_term = self.kd * derivative

        # Update state
        self.prev_error = error

        # PID output
        return current_lambda + p_term + i_term + d_term

    def reset(self) -> None:
        """Reset internal state (useful between experiments)."""
        self.velocity = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
