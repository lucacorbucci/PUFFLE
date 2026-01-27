# ABOUTME: Provides utility functions for computing demographic disparity and fairness metrics.
# ABOUTME: Includes both standard and differentiable implementations of group fairness metrics.

import torch


def compute_demographic_disparity(
    z: torch.Tensor,
    y: torch.Tensor,
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
        msg = f"At least two unique values for the sensitive attribute z are required to compute disparity. Only {num_z} found."
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
    p_y_given_z = pair_counts / (z_counts.view(-1, 1) + 1e-10)

    # 2. Compute P(Y=y | Z!=z) for all y, z
    # Total count of y across the whole dataset
    y_counts = torch.bincount(y_inverse, minlength=num_y).float()  # [y]

    # count(Z!=z, Y=y) = count(Y=y) - count(Z=z, Y=y)
    count_not_z_y = y_counts.view(1, -1) - pair_counts

    # count(Z!=z) = total_samples - count(Z=z)
    total_samples = len(z)
    count_not_z = total_samples - z_counts

    p_y_given_not_z = count_not_z_y / (count_not_z.view(-1, 1) + 1e-10)

    # 3. Compute disparity |P(Y=y|Z=z) - P(Y=y|Z!=z)|
    disparities = torch.abs(p_y_given_z - p_y_given_not_z)
    max_disparity = disparities.max().item()

    # Return empty statistics to match type signature.
    statistics = {
        "counter_z": 0,
        "counter_not_z": 0,
        "counter_y_z": 0,
        "counter_y_not_z": 0,
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

    unique_sensitive_attributes = torch.unique(sensitive_attributes)
    unique_targets = torch.unique(predictions_argmax)

    if len(unique_sensitive_attributes) == 0 or len(unique_targets) == 0:
        msg = "Input tensors sensitive_attributes and predictions_argmax must not be empty."
        raise ValueError(msg)
    if len(unique_sensitive_attributes) == 1 or len(unique_targets) == 1:
        msg = "Input tensors sensitive_attributes and predictions_argmax must have more than one unique value."
        raise ValueError(msg)

    fairness_violations = []
    for target in unique_targets:
        for sensitive_attribute in unique_sensitive_attributes:
            # We get the number of samples that are predicted with the target class
            # target and that have the sensitive attribute equal to z:  |Y = k, Z = z|.
            # In this case we just sum the columns of the rows that
            # respect the previous constraint.
            # Example: Given [[0.2, 0.8], [0.4, 0.6], [0.3, 0.7]], suppose
            # that to compute Y_eq_k_and_Z_eq_z we have to consider only
            # the first and the third row and that we are considering the class 1.
            # In this case we will sum 0.8 and 0.7.
            y_eq_k_and_z_eq_z = torch.sum(
                softmax_output[
                    (predictions_argmax == target)
                    & (sensitive_attributes == sensitive_attribute)
                ][:, target]
            )

            # Here we compute |Y = k, Z != z| with the same strategy we used to
            # compute |Y = k, Z = z|.
            y_eq_k_and_z_not_eq_z = torch.sum(
                softmax_output[
                    (predictions_argmax == target)
                    & (sensitive_attributes != sensitive_attribute)
                ][:, target]
            )

            z_eq_z = torch.sum(
                softmax_output[(sensitive_attributes == sensitive_attribute)][:, target]
            )

            z_not_eq_z = torch.sum(
                softmax_output[(sensitive_attributes != sensitive_attribute)][:, target]
            )
            # TODO: we need to check if z_eq_z and z_not_eq_z are not equal to 0
            # if they are equal to 0 we need to use the information present in the
            # probabilities dictionary, if the probabilities dictionary is None
            # then we need to raise an error
            violation_term = torch.abs(
                (y_eq_k_and_z_eq_z / z_eq_z) - (y_eq_k_and_z_not_eq_z / z_not_eq_z)
            )
            fairness_violations.append(violation_term)

    fairness_violations = torch.stack(fairness_violations)
    max_violation, _ = torch.max(fairness_violations, dim=0)
    return max_violation
