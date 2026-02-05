import torch

from puffle.Utils.metric_utils import compute_binary_statistics
from puffle.Utils.privacy import get_noise


def _compute_p_y_given_group_with_fallback(
    numerator_matrix: torch.Tensor,
    denominator_vector: torch.Tensor,
    unique_z: torch.Tensor,
    unique_y: torch.Tensor,
    average_probabilities: dict | None,
    *,
    is_complement: bool,
) -> torch.Tensor:
    if average_probabilities is None:
        return (numerator_matrix) / (denominator_vector.view(-1, 1) + 1e-10)

    probs = torch.zeros_like(numerator_matrix, dtype=torch.float)
    for z_idx, z_val in enumerate(unique_z):
        den = denominator_vector[z_idx]
        if den > 0:
            for y_idx, _ in enumerate(unique_y):
                n_val = numerator_matrix[z_idx, y_idx]
                d_val = den

                probs[z_idx, y_idx] = (n_val) / (d_val + 1e-10)
        else:
            other_z_val = 1 - int(z_val.item()) if is_complement else int(z_val.item())
            for y_idx, y_val in enumerate(unique_y):
                key = f"{int(y_val.item())}|{other_z_val}"
                probs[z_idx, y_idx] = average_probabilities.get(key, 0.0)
    return probs


def compute_demographic_disparity(
    z: torch.Tensor,
    y: torch.Tensor,
    average_probabilities: dict | None = None,
    is_validation: bool = False,  # noqa: FBT001, FBT002
    sigma_update_lambda: float | None = None,
):
    """
    Compute the demographic disparity of a model.
    Defined as:
    max_{z, y} |P(Y=y|Z=z) - P(Y=y|Z!=z)|
    where P(Y=y|Z=z) is the probability of the target value y
    given the sensitive feature z.

    Args:
        z (torch.Tensor): The sensitive features.
        y (torch.Tensor): The target values.
        average_probabilities (dict | None): Global statistics for fallback.
        is_validation (bool): Whether this is validation mode.
        sigma_update_lambda (float | None): Noise parameter for differential privacy.

    Returns:
        tuple[float, dict]: The demographic disparity of the model and a dictionary of statistics.

    """
    if len(z) != len(y):
        msg = "Input tensors z and y must have the same length."
        raise ValueError(msg)
    if not isinstance(z, torch.Tensor) or not isinstance(y, torch.Tensor):
        msg = "Input tensors z and y must be of type torch.Tensor."
        raise TypeError(msg)

    if len(z) == 0:
        msg = "Input tensors z and y must not be empty."
        raise ValueError(msg)

    unique_z, z_inverse = torch.unique(z, return_inverse=True)
    unique_y, y_inverse = torch.unique(y, return_inverse=True)

    num_z = len(unique_z)
    num_y = len(unique_y)

    # 1. Compute P(Y=y | Z=z)
    pair_indices = z_inverse * num_y + y_inverse
    pair_counts = torch.bincount(pair_indices, minlength=num_z * num_y).float()
    pair_counts = pair_counts.view(num_z, num_y)  # [z, y]
    z_counts = torch.bincount(z_inverse, minlength=num_z).float()  # [z]

    # 2. Compute P(Y=y | Z!=z)
    y_counts = torch.bincount(y_inverse, minlength=num_y).float()  # [y]
    count_not_z_y = y_counts.view(1, -1) - pair_counts
    total_samples = len(z)
    count_not_z = total_samples - z_counts

    if (
        average_probabilities is not None and average_probabilities.get("first_round")
    ) or is_validation:
        max_disparity = 0.0
        # Compute statistics for FL aggregation with validation
        # Validation only makes sense if we can uniquely identify binary groups 0 and 1
        statistics = compute_binary_statistics(
            num_z,
            unique_z,
            unique_y,
            z_counts,
            pair_counts,
            total_samples,
            z,
            y,
        )
        return max_disparity, statistics

    min_required_groups = 2
    if num_z < min_required_groups and average_probabilities is None:
        msg = f"At least two unique values for the sensitive attribute z are required to compute disparity. Only {num_z} found. {z}"
        raise ValueError(msg)

    if sigma_update_lambda is not None:
        pair_counts += get_noise(
            mechanism_type="gaussian",
            sigma=sigma_update_lambda,
        )
        z_counts += get_noise(
            mechanism_type="gaussian",
            sigma=sigma_update_lambda,
        )
        count_not_z_y += get_noise(
            mechanism_type="gaussian",
            sigma=sigma_update_lambda,
        )
        count_not_z += get_noise(
            mechanism_type="gaussian",
            sigma=sigma_update_lambda,
        )

    # Compute the two conditional probabilities
    p_y_given_z = _compute_p_y_given_group_with_fallback(
        pair_counts,
        z_counts,
        unique_z,
        unique_y,
        average_probabilities,
        is_complement=False,
    )

    p_y_given_not_z = _compute_p_y_given_group_with_fallback(
        count_not_z_y,
        count_not_z,
        unique_z,
        unique_y,
        average_probabilities,
        is_complement=True,
    )

    # 3. Compute disparity
    disparities = torch.abs(p_y_given_z - p_y_given_not_z)
    max_disparity = disparities.max().item()

    # Compute statistics for FL aggregation with validation
    # Validation only makes sense if we can uniquely identify binary groups 0 and 1
    statistics = compute_binary_statistics(
        num_z,
        unique_z,
        unique_y,
        z_counts,
        pair_counts,
        total_samples,
        z,
        y,
    )

    return max_disparity, statistics


def compute_differentiable_demographic_disparity(
    predictions_argmax: torch.Tensor,
    sensitive_attributes: torch.Tensor,
    softmax_output: torch.Tensor,
):
    """
    Compute the demographic disparity of a model in a differentiable way.
    The demographic disparity is defined as:
    max_{z, y} |P(Y=y|Z=z) - P(Y=y|Z!=z)|
    where P(Y=y|Z=z) is the probability of the target value y
    given the sensitive feature z.

    In this case, we use the softmax output of the model to compute
    the demographic disparity so that the result is differentiable.

    Args:
        predictions_argmax (torch.Tensor): The predicted classes.
        sensitive_attributes (torch.Tensor): The sensitive attributes.
        softmax_output (torch.Tensor): The softmax output of the model.

    Returns:
        torch.Tensor: The demographic disparity of the model.

    """
    if len(sensitive_attributes) != len(predictions_argmax):
        msg = "Input tensors sensitive_attributes and predictions_argmax must have the same length."
        raise ValueError(msg)
    if not isinstance(sensitive_attributes, torch.Tensor) or not isinstance(
        predictions_argmax, torch.Tensor
    ):
        msg = "Input tensors sensitive_attributes and predictions_argmax must be of type torch.Tensor."
        raise TypeError(msg)

    unique_z = torch.unique(sensitive_attributes)
    unique_y = torch.unique(predictions_argmax)

    if len(unique_z) == 0 or len(unique_y) == 0:
        msg = "Input tensors sensitive_attributes and predictions_argmax must not be empty."
        raise ValueError(msg)
    if len(unique_z) == 1 or len(unique_y) == 1:
        msg = "Input tensors sensitive_attributes and predictions_argmax must have more than one unique value."
        raise ValueError(msg)

    # Vectorized computation using broadcasting
    # unique_y: [Y], unique_z: [Z]
    # predictions_argmax: [N], sensitive_attributes: [N], softmax_output: [N, C]

    # Create masks: [Y, N] and [Z, N]
    y_mask = (predictions_argmax.unsqueeze(0) == unique_y.unsqueeze(1)).float()
    z_mask = (sensitive_attributes.unsqueeze(0) == unique_z.unsqueeze(1)).float()

    # Combined mask: [Y, Z, N]
    joint_mask = y_mask.unsqueeze(1) * z_mask.unsqueeze(0)

    # Extract relevant softmax columns for each unique target: [Y, N]
    # We only care about columns corresponding to unique_y
    softmax_y = softmax_output[:, unique_y].t()

    # Numerators: sum(softmax * joint_mask) for each y in Y, z in Z: [Y, Z]
    # softmax_y.unsqueeze(1): [Y, 1, N], joint_mask: [Y, Z, N]
    numerator = torch.sum(softmax_y.unsqueeze(1) * joint_mask, dim=2)

    # Denominators: sum(softmax * z_mask) for each y in Y, z in Z: [Y, Z]
    # softmax_y.unsqueeze(1): [Y, 1, N], z_mask.unsqueeze(0): [1, Z, N]
    denominator = torch.sum(softmax_y.unsqueeze(1) * z_mask.unsqueeze(0), dim=2)

    # P(Y=k|Z=z)
    p_k_z = numerator / (denominator + 1e-10)

    # For Z != z:
    # total_k = sum(softmax * y_mask) for each k in Y: [Y]
    total_k = torch.sum(softmax_y * y_mask, dim=1)
    numerator_not_z = total_k.unsqueeze(1) - numerator

    # total_den_k = sum(softmax) for each k in Y: [Y]
    total_den_k = torch.sum(softmax_y, dim=1)
    denominator_not_z = total_den_k.unsqueeze(1) - denominator

    p_k_not_z = numerator_not_z / (denominator_not_z + 1e-10)

    violation = torch.abs(p_k_z - p_k_not_z)
    return violation.max()
