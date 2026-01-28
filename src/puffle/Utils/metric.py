import torch

from puffle.Utils.privacy import get_noise


def compute_demographic_disparity(
    z: torch.Tensor,
    y: torch.Tensor,
    sigma_update_lambda: float = None,
    average_probabilities: dict = None,
):
    """
    Compute the demographic disparity of a model.
    Defined as:
    max_{z, y} |P(Y=y|Z=z) - P(Y=y|Z!=z)|
    where P(Y=y|Z=z) is the probability of the target value y
    given the sensitive feature z.

    Args:
        x (torch.Tensor): The input features.
        z (torch.Tensor): The sensitive features.
        y (torch.Tensor): The target values.

    Returns:
        tuple[float, dict]: The demographic disparity of the model and a dictionary of statistics
        (empty in the current vectorized implementation).

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

    min_required_groups = 2
    if num_z < min_required_groups:
        msg = f"At least two unique values for the sensitive attribute z are required to compute disparity. Only {num_z} found. {z}"
        raise ValueError(msg)

    # 1. Compute P(Y=y | Z=z) for all y, z
    # We can use bincount on pairs.
    # Map (z, y) pairs to unique indices: index = z_idx * num_y + y_idx
    pair_indices = z_inverse * num_y + y_inverse
    pair_counts = torch.bincount(pair_indices, minlength=num_z * num_y).float()
    pair_counts = pair_counts.view(num_z, num_y)  # [z, y]

    z_counts = torch.bincount(z_inverse, minlength=num_z).float()  # [z]

    # Probabilities P(Y=y | Z=z) = count(z,y) / count(z)
    # Avoid division by zero
    if average_probabilities is None:
        p_y_given_z = (
            pair_counts
            + (
                get_noise(
                    mechanism_type="gaussian",
                    sigma=sigma_update_lambda,
                )
                if sigma_update_lambda is not None
                else 0
            )
        ) / (
            z_counts.view(-1, 1)
            + 1e-10
            + +(
                get_noise(
                    mechanism_type="gaussian",
                    sigma=sigma_update_lambda,
                )
                if sigma_update_lambda is not None
                else 0
            )
        )
    else:
        p_y_given_z = torch.zeros_like(pair_counts, dtype=torch.float)

        for z_idx, z_val in enumerate(unique_z):
            z_denominator = z_counts[z_idx]

            # Check if this group has samples (denominator > 0)
            if z_denominator > 0:
                # Use local computation
                for y_idx, y_val in enumerate(unique_y):
                    numerator = pair_counts[z_idx, y_idx] + (
                        get_noise(
                            mechanism_type="gaussian",
                            sigma=sigma_update_lambda,
                        )
                        if sigma_update_lambda is not None
                        else 0
                    )
                    denominator = z_denominator + (
                        get_noise(
                            mechanism_type="gaussian",
                            sigma=sigma_update_lambda,
                        )
                        if sigma_update_lambda is not None
                        else 0
                    )
                    p_y_given_z[z_idx, y_idx] = numerator / (denominator + 1e-10)
            else:
                # Use global average probabilities as fallback
                for y_idx, y_val in enumerate(unique_y):
                    key = f"{int(y_val.item())}|{int(z_val.item())}"
                    if key in average_probabilities:
                        p_y_given_z[z_idx, y_idx] = average_probabilities[key]
                    else:
                        raise ValueError(
                            f"Key {key} not found in average probabilities."
                        )

    # 2. Compute P(Y=y | Z!=z) for all y, z
    # Total count of y across the whole dataset
    y_counts = torch.bincount(y_inverse, minlength=num_y).float()  # [y]

    # count(Z!=z, Y=y) = count(Y=y) - count(Z=z, Y=y)
    count_not_z_y = y_counts.view(1, -1) - pair_counts

    # count(Z!=z) = total_samples - count(Z=z)
    total_samples = len(z)
    count_not_z = total_samples - z_counts

    if average_probabilities is None:
        p_y_given_not_z = (
            count_not_z_y
            + +(
                get_noise(
                    mechanism_type="gaussian",
                    sigma=sigma_update_lambda,
                )
                if sigma_update_lambda is not None
                else 0
            )
        ) / (
            count_not_z.view(-1, 1)
            + 1e-10
            + +(
                get_noise(
                    mechanism_type="gaussian",
                    sigma=sigma_update_lambda,
                )
                if sigma_update_lambda is not None
                else 0
            )
        )
    else:
        p_y_given_not_z = torch.zeros_like(count_not_z_y, dtype=torch.float)

        for z_idx, z_val in enumerate(unique_z):
            not_z_denominator = count_not_z[z_idx]

            if not_z_denominator > 0:
                for y_idx, y_val in enumerate(unique_y):
                    numerator = count_not_z_y[z_idx, y_idx] + (
                        get_noise(
                            mechanism_type="gaussian",
                            sigma=sigma_update_lambda,
                        )
                        if sigma_update_lambda is not None
                        else 0
                    )
                    denominator = not_z_denominator + (
                        get_noise(
                            mechanism_type="gaussian",
                            sigma=sigma_update_lambda,
                        )
                        if sigma_update_lambda is not None
                        else 0
                    )
                    p_y_given_not_z[z_idx, y_idx] = numerator / (denominator + 1e-10)
            else:
                # Fallback: We need P(Y|NOT Z).
                # Since we only support binary sensitive attribute for now in average_probabilities logic (implied by "1|0"),
                # "NOT Z" corresponds to the other z value.
                other_z_val = 1 - int(z_val.item())  # Assuming 0/1 coding
                for y_idx, y_val in enumerate(unique_y):
                    key = f"{int(y_val.item())}|{other_z_val}"
                    if key in average_probabilities:
                        p_y_given_not_z[z_idx, y_idx] = average_probabilities[key]
                    else:
                        p_y_given_not_z[z_idx, y_idx] = 0.0

    # 3. Compute disparity |P(Y=y|Z=z) - P(Y=y|Z!=z)|
    disparities = torch.abs(p_y_given_z - p_y_given_not_z)
    max_disparity = disparities.max().item()

    # Compute statistics for FL aggregation
    # Assuming binary z (0/1) and y (0/1), we extract counts for the positive class
    # counter_z: count of samples where z == 1 (or the second unique z value)
    # counter_not_z: count of samples where z == 0 (or the first unique z value)
    # counter_y_z: count of samples where y == 1 AND z == 1
    # counter_y_not_z: count of samples where y == 1 AND z == 0
    if num_z >= min_required_groups and num_y >= min_required_groups:
        # z_counts[0] = count of first unique z value (typically 0)
        # z_counts[1] = count of second unique z value (typically 1)
        counter_z = int(z_counts[1].item()) if num_z > 1 else 0
        counter_not_z = int(z_counts[0].item())
        # pair_counts[z_idx, y_idx] = count of (z, y) pairs
        # pair_counts[1, 1] = count of (z=1, y=1)
        # pair_counts[0, 1] = count of (z=0, y=1)
        counter_y_z = int(pair_counts[1, 1].item()) if num_z > 1 and num_y > 1 else 0
        counter_y_not_z = int(pair_counts[0, 1].item()) if num_y > 1 else 0
    else:
        counter_z = 0
        counter_not_z = 0
        counter_y_z = 0
        counter_y_not_z = 0

    statistics = {
        "counter_z": counter_z,
        "counter_not_z": counter_not_z,
        "counter_y_z": counter_y_z,
        "counter_y_not_z": counter_y_not_z,
    }

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
        probabilities (dict, optional): A dictionary containing the probabilities
            of the target values. This is used in FL in the cases in which the client
            does not have all the possible classes/sensitive values. Defaults to None.

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
