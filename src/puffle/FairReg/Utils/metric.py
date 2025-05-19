import numpy as np
import torch
from opacus.optimizers.optimizer import DPOptimizer


def compute_demographic_disparity(
    z: torch.Tensor,
    y: torch.Tensor,
):
    """
    Compute the demographic disparity of a model.
    The demographic disparity is defined as:
    max_{z, y} |P(Y=y|Z=z) - P(Y=y|Z!=z)|
    where P(Y=y|Z=z) is the probability of the target value y
    given the sensitive feature z.

    Args:
        x (torch.Tensor): The input features.
        z (torch.Tensor): The sensitive features.
        y (torch.Tensor): The target values.

    Returns:
        float: The demographic disparity of the model.
    """
    if len(z) != len(y):
        raise ValueError("Input tensors z and y must have the same length.")
    if not isinstance(z, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("Input tensors z and y must be of type torch.Tensor.")

    unique_z = torch.unique(z)
    unique_y = torch.unique(y)

    if len(unique_z) == 0 or len(unique_y) == 0:
        raise ValueError("Input tensors z and y must not be empty.")
    if len(unique_z) == 1 or len(unique_y) == 1:
        raise ValueError("Input tensors z and y must have more than one unique value.")

    max_disparity = 0

    for z_val in unique_z:
        for y_val in unique_y:
            # Compute the probability of y given z
            p_y_given_z = (y[(z == z_val)] == y_val).float().mean().item()
            # Compute the probability of y given not z
            p_y_given_not_z = (y[(z != z_val)] == y_val).float().mean().item()
            # Compute the absolute difference
            disparity = abs(p_y_given_z - p_y_given_not_z)

            # Update the maximum disparity
            max_disparity = max(max_disparity, disparity)

    return max_disparity


def compute_differentiable_demographic_disparity(
    predictions_argmax: torch.Tensor,
    sensitive_attributes: torch.Tensor,
    softmax_output: torch.Tensor,
    probabilities: dict = None,
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
        raise ValueError("Input tensors sensitive_attributes and predictions_argmax must have the same length.")
    if not isinstance(sensitive_attributes, torch.Tensor) or not isinstance(predictions_argmax, torch.Tensor):
        raise TypeError("Input tensors sensitive_attributes and predictions_argmax must be of type torch.Tensor.")

    unique_sensitive_attributes = torch.unique(sensitive_attributes)
    unique_targets = torch.unique(predictions_argmax)

    if len(unique_sensitive_attributes) == 0 or len(unique_targets) == 0:
        raise ValueError("Input tensors sensitive_attributes and predictions_argmax must not be empty.")
    if len(unique_sensitive_attributes) == 1 or len(unique_targets) == 1:
        raise ValueError("Input tensors sensitive_attributes and predictions_argmax must have more than one unique value.")

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
            Y_eq_k_and_Z_eq_z = torch.sum(
                softmax_output[(predictions_argmax == target) & (sensitive_attributes == sensitive_attribute)][:, target]
            )

            # Here we compute |Y = k, Z != z| with the same strategy we used to
            # compute |Y = k, Z = z|.
            Y_eq_k_and_Z_not_eq_z = torch.sum(
                softmax_output[(predictions_argmax == target) & (sensitive_attributes != sensitive_attribute)][:, target]
            )
            
            Z_eq_z = torch.sum(
                softmax_output[(sensitive_attributes == sensitive_attribute)][:, target]
            )

            Z_not_eq_z = torch.sum(
                softmax_output[(sensitive_attributes != sensitive_attribute)][:, target]
            )
            # todo: we need to check if Z_eq_z and Z_not_eq_z are not equal to 0
            # if they are equal to 0 we need to use the information present in the 
            # probabilities dictionary, if the probabilities dictionary is None
            # then we need to raise an error
            violation_term = torch.abs((Y_eq_k_and_Z_eq_z / Z_eq_z) - (Y_eq_k_and_Z_not_eq_z / Z_not_eq_z))
            fairness_violations.append(violation_term)
    
    
    
    fairness_violations = torch.stack(fairness_violations)
    max_violation, _ = torch.max(fairness_violations, dim=0)
    return max_violation