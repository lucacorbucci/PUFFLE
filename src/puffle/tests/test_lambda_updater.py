import pytest

from puffle.Utils.lambda_updater import LambdaUpdater, LambdaUpdateStrategy


def test_lambda_updater_momentum():
    updater = LambdaUpdater(
        strategy=LambdaUpdateStrategy.MOMENTUM, alpha=0.1, momentum=0.9
    )
    # unfairness > target => delta = target - unfairness is negative
    # current_lambda - velocity * alpha
    current_lambda = 0.5
    unfairness = 0.6
    target = 0.1
    # delta = 0.1 - 0.6 = -0.5
    # velocity = 0.9 * 0 + (-0.5) = -0.5
    # new_lambda = 0.5 - (-0.5 * 0.1) = 0.5 + 0.05 = 0.55
    new_lambda = updater.update(current_lambda, unfairness, target)
    assert new_lambda == pytest.approx(0.55)

    assert updater.velocity == pytest.approx(-0.5)


def test_lambda_updater_pid():
    updater = LambdaUpdater(strategy=LambdaUpdateStrategy.PID, kp=0.1, ki=0.01, kd=0.05)
    current_lambda = 0.5
    unfairness = 0.6
    target = 0.1
    # error = 0.6 - 0.1 = 0.5
    # p_term = 0.1 * 0.5 = 0.05
    # integral = 0 + 0.5 = 0.5
    # i_term = 0.01 * 0.5 = 0.005
    # derivative = 0.5 - 0 = 0.5
    # d_term = 0.05 * 0.5 = 0.025
    # new_lambda = 0.5 + 0.05 + 0.005 + 0.025 = 0.58
    new_lambda = updater.update(current_lambda, unfairness, target)
    assert new_lambda == pytest.approx(0.58)
    assert updater.integral == pytest.approx(0.5)
    assert updater.prev_error == pytest.approx(0.5)


def test_lambda_updater_reset():
    updater = LambdaUpdater(strategy=LambdaUpdateStrategy.MOMENTUM)
    updater.velocity = 1.0
    updater.integral = 1.0
    updater.prev_error = 1.0

    updater.reset()
    assert updater.velocity == 0.0
    assert updater.integral == 0.0
    assert updater.prev_error == 0.0


def test_lambda_updater_invalid_strategy():
    updater = LambdaUpdater(strategy="invalid")  # type: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="Unknown strategy"):
        updater.update(0.5, 0.6, 0.1)


def test_lambda_updater_constraints():
    updater = LambdaUpdater(strategy=LambdaUpdateStrategy.GRADIENT, alpha=10.0)
    # Force out of bounds
    # 0.5 + 10.0 * (0.9 - 0.1) = 0.5 + 8.0 = 8.5 => 1.0
    new_lambda = updater.update(0.5, 0.9, 0.1)
    assert new_lambda == 1.0

    # 0.5 + 10.0 * (0.1 - 0.9) = 0.5 - 8.0 = -7.5 => 0.0
    new_lambda = updater.update(0.5, 0.1, 0.9)
    assert new_lambda == 0.0
