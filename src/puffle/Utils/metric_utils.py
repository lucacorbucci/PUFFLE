import logging
from logging import INFO

import torch
from flwr.common.logger import log

logger = logging.getLogger(__name__)

BINARY_CLASS_COUNT = 2


def compute_binary_statistics(
    num_z, unique_z, unique_y, z_counts, pair_counts, total_samples, z, y
):
    """
    Compute and validate all statistics for binary sensitive attribute case.

    Returns a dictionary with all counters for the 2x2 contingency table:
    {
        'counter_z': samples with Z=1,
        'counter_not_z': samples with Z=0,
        'counter_y': samples with Y=1,
        'counter_not_y': samples with Y=0,
        'counter_y_z': samples with Z=1 AND Y=1,
        'counter_y_not_z': samples with Z=0 AND Y=1,
        'counter_not_y_z': samples with Z=1 AND Y=0,
        'counter_not_y_not_z': samples with Z=0 AND Y=0,
        'total_samples': total number of samples
    }

    Returns dict with all zeros if validation fails.
    """
    # Initialize empty result
    empty_result = {
        "counter_z": 0,
        "counter_not_z": 0,
        "counter_y": 0,
        "counter_not_y": 0,
        "counter_y_z": 0,
        "counter_y_not_z": 0,
        "counter_not_y_z": 0,
        "counter_not_y_not_z": 0,
        "total_samples": 0,
    }

    # Initial validation checks
    # 1. Must have at least one Z group
    # 2. Z must be strictly [0, 1]
    # 3. Y values must be within binary range [0, 1]
    # Check if Z values are valid (subset of [0, 1])
    z_is_binary = ((unique_z == 0) | (unique_z == 1)).all()
    
    is_valid_input = (
        num_z >= 1 and
        z_is_binary and
        not ((unique_y < 0).any() or (unique_y > 1).any())
    )
    
    if not is_valid_input:
        return empty_result

    # Find indices for Z=1, Z=0 in unique_z
    z_pos_idx = (unique_z == 1).nonzero(as_tuple=True)[0]
    z_neg_idx = (unique_z == 0).nonzero(as_tuple=True)[0]

    # Verify pair_counts shape matches num_z
    if pair_counts.shape[0] != num_z:
        return empty_result

    # Extract all counters
    # Marginal counts for Z
    counter_z = int(z_counts[z_pos_idx[0]].item()) if len(z_pos_idx) > 0 else 0
    counter_not_z = int(z_counts[z_neg_idx[0]].item()) if len(z_neg_idx) > 0 else 0

    # Initialize Y-related counters to 0
    counter_y = 0
    counter_not_y = 0
    counter_y_z = 0
    counter_y_not_z = 0
    counter_not_y_z = 0
    counter_not_y_not_z = 0

    y_pos_idx = (unique_y == 1).nonzero(as_tuple=True)[0]
    y_neg_idx = (unique_y == 0).nonzero(as_tuple=True)[0]

    # Handle Y=1 counters (if Y=1 exists)
    if len(y_pos_idx) > 0:
        idx = y_pos_idx[0]
        if pair_counts.shape[1] > idx:
            # col sum is always safe
            counter_y = int(pair_counts[:, idx].sum().item())
            # Z=1, Y=1
            if len(z_pos_idx) > 0:
                counter_y_z = int(pair_counts[z_pos_idx[0], idx].item())
            # Z=0, Y=1
            if len(z_neg_idx) > 0:
                counter_y_not_z = int(pair_counts[z_neg_idx[0], idx].item())

    # Handle Y=0 counters (if Y=0 exists)
    if len(y_neg_idx) > 0:
        idx = y_neg_idx[0]
        if pair_counts.shape[1] > idx:
            counter_not_y = int(pair_counts[:, idx].sum().item())
            # Z=1, Y=0
            if len(z_pos_idx) > 0:
                counter_not_y_z = int(pair_counts[z_pos_idx[0], idx].item())
            # Z=0, Y=0
            if len(z_neg_idx) > 0:
                counter_not_y_not_z = int(pair_counts[z_neg_idx[0], idx].item())

    # Create result dictionary
    result = {
        "counter_z": counter_z,
        "counter_not_z": counter_not_z,
        "counter_y": counter_y,
        "counter_not_y": counter_not_y,
        "counter_y_z": counter_y_z,
        "counter_y_not_z": counter_y_not_z,
        "counter_not_y_z": counter_not_y_z,
        "counter_not_y_not_z": counter_not_y_not_z,
        "total_samples": total_samples,
    }


    # Validation
    if not _validate_counters(result, total_samples, z, y):
        return empty_result

    return result


def _validate_counters(result, total_samples, z, y):
    """Perform all validation checks on computed counters."""
    if not _validate_marginal_sums(result, total_samples):
        return False

    if not _validate_joint_sums(result, total_samples):
        return False

    if not _validate_row_col_consistency(result):
        return False

    # Validation 6: All counts should be non-negative
    if any(v < 0 for v in result.values()):
        logger.warning("Negative counts detected")
        return False

    # Validation 7: Cross-check with actual data
    try:
        validate_all_counters(z, y, result)
    except AssertionError as e:
        logger.warning("Counter validation failed: %s", e)
        return False

    return True


def _validate_marginal_sums(result, total_samples):
    """Validate marginal Z and Y sums."""
    if result["counter_z"] + result["counter_not_z"] != total_samples:
        logger.warning(
            "Z counts mismatch: %s + %s != %s",
            result["counter_z"],
            result["counter_not_z"],
            total_samples,
        )
        return False

    if result["counter_y"] + result["counter_not_y"] != total_samples:
        logger.warning(
            "Y counts mismatch: %s + %s != %s",
            result["counter_y"],
            result["counter_not_y"],
            total_samples,
        )
        return False
    return True


def _validate_joint_sums(result, total_samples):
    """Validate joint counts sum to total."""
    sum_joint = (
        result["counter_y_z"]
        + result["counter_y_not_z"]
        + result["counter_not_y_z"]
        + result["counter_not_y_not_z"]
    )
    if sum_joint != total_samples:
        logger.warning(
            "Joint counts don't sum to total: %s != %s", sum_joint, total_samples
        )
        return False
    return True


def _validate_row_col_consistency(result):
    """Validate row and column sums against marginals."""
    # Row sums (Z)
    if result["counter_y_z"] + result["counter_not_y_z"] != result["counter_z"]:
        logger.warning(
            "Row sum mismatch for Z=1: %s + %s != %s",
            result["counter_y_z"],
            result["counter_not_y_z"],
            result["counter_z"],
        )
        return False

    if (
        result["counter_y_not_z"] + result["counter_not_y_not_z"]
        != result["counter_not_z"]
    ):
        logger.warning(
            "Row sum mismatch for Z=0: %s + %s != %s",
            result["counter_y_not_z"],
            result["counter_not_y_not_z"],
            result["counter_not_z"],
        )
        return False

    # Column sums (Y)
    if result["counter_y_z"] + result["counter_y_not_z"] != result["counter_y"]:
        logger.warning(
            "Column sum mismatch for Y=1: %s + %s != %s",
            result["counter_y_z"],
            result["counter_y_not_z"],
            result["counter_y"],
        )
        return False

    if (
        result["counter_not_y_z"] + result["counter_not_y_not_z"]
        != result["counter_not_y"]
    ):
        logger.warning(
            "Column sum mismatch for Y=0: %s + %s != %s",
            result["counter_not_y_z"],
            result["counter_not_y_not_z"],
            result["counter_not_y"],
        )
        return False
    return True


def validate_all_counters(z, y, counters):
    """Validate that all computed counters match actual data."""
    z_eq_1 = z == 1
    z_eq_0 = z == 0
    y_eq_1 = y == 1
    y_eq_0 = y == 0

    # Compute actual counts from raw data
    actual = {
        "counter_z": z_eq_1.sum().item(),
        "counter_not_z": z_eq_0.sum().item(),
        "counter_y": y_eq_1.sum().item(),
        "counter_not_y": y_eq_0.sum().item(),
        "counter_y_z": (z_eq_1 & y_eq_1).sum().item(),
        "counter_y_not_z": (z_eq_0 & y_eq_1).sum().item(),
        "counter_not_y_z": (z_eq_1 & y_eq_0).sum().item(),
        "counter_not_y_not_z": (z_eq_0 & y_eq_0).sum().item(),
        "total_samples": len(z),
    }

    # Compare each counter using .items()
    for name, expected in actual.items():
        computed = counters[name]
        if computed != expected:
            msg = f"{name} mismatch: computed={computed}, actual={expected}"
            raise AssertionError(msg)

    return True
