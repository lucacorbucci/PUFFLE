import numpy as np
import torch
from opacus.optimizers.optimizer import DPOptimizer


class Utils:
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
            return (
                np.random.geometric(p=p, size=1) - np.random.geometric(p=p, size=1)
            )[0]
        elif mechanism_type == "gaussian":
            return np.random.normal(loc=0, scale=sigma, size=1)[0]
        else:
            raise ValueError(
                "The mechanism type must be either laplace, geometric or gaussian"
            )

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
        per_param_norms = [
            g.reshape(len(g), -1).norm(2, dim=-1) for g in optimizer.grad_samples
        ]
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
