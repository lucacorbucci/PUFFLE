import os
import random

import numpy as np
import torch
from opacus.optimizers.optimizer import DPOptimizer


class Utils:
    @staticmethod
    def compute_demographic_disparity(
        x: torch.Tensor,
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
        unique_z = torch.unique(z)
        unique_y = torch.unique(y)

        max_disparity = 0

        for z_val in unique_z:
            for y_val in unique_y:
                # Compute the probability of y given z
                p_y_given_z = torch.mean(y[z == z_val] == y_val).item()

                # Compute the probability of y given not z
                p_y_given_not_z = torch.mean(y[z != z_val] == y_val).item()

                # Compute the absolute difference
                disparity = torch.abs(p_y_given_z - p_y_given_not_z)

                # Update the maximum disparity
                max_disparity = max(max_disparity, disparity)

        return max_disparity

    def compute_differentiable_demographic_disparity(
        x: torch.Tensor,
        z: torch.Tensor,
        y: torch.Tensor,
        softmax_output: torch.Tensor,
    ):
        pass

    @staticmethod
    def get_noise(
        mechanism_type: str,
        epsilon: float = None,
        sensitivity: float = None,
        sigma: float = None,
    ):
        if mechanism_type == "laplace":
            return np.random.laplace(loc=0, scale=sensitivity / epsilon, size=1)
        elif mechanism_type == "geometric":
            p = 1 - np.exp(-epsilon / sensitivity)
            return (np.random.geometric(p=p, size=1) - np.random.geometric(p=p, size=1))[0]
        elif mechanism_type == "gaussian":
            return np.random.normal(loc=0, scale=sigma, size=1)[0]
        else:
            raise ValueError("The mechanism type must be either laplace, geometric or gaussian")

    @staticmethod
    def get_summed_grad(model, batch_size):
        # Compute the 2-norm of the gradients
        total_norm = 0
        for p in model.parameters():
            param_norm = p.summed_grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        return total_norm / batch_size

    @staticmethod
    def compute_gradient_norm(model: torch.nn.Module):
        # Compute the 2-norm of the gradients
        total_norm = 0
        for p in model.parameters():
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5

        return total_norm

    @staticmethod
    def compute_per_sample_gradient_norm(model: torch.nn.Module, batch_size: int):
        # Compute the 2-norm of the gradients
        total_norm = 0
        for p in model.parameters():
            tmp_norm = 0
            for i in range(batch_size):
                param_norm = p.grad_sample[i].detach().data.norm(2)
                tmp_norm += param_norm.item() ** 2
            total_norm = tmp_norm / batch_size
        total_norm = total_norm**0.5

        return total_norm

    @staticmethod
    def compute_max_and_min_per_sample_gradient(optimizer: DPOptimizer):
        per_param_norms = [g.reshape(len(g), -1).norm(2, dim=-1) for g in optimizer.grad_samples]
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
        return (
            min(per_sample_norms),
            max(per_sample_norms),
            torch.mean(per_sample_norms),
        )

    @staticmethod
    def sync_models(model_1, model_2):
        """Sync the parameters of two models
        so that they have the same values.

        Args:
            model_1 (_type_): the first model
            model_2 (_type_): the second model
        """
        for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
            p1.data = p2.data.clone()

        for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
            assert torch.all(p1 == p2)

    @staticmethod
    def seed_everything(seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
