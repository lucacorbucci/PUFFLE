from typing import Tuple

import torch
from Models.celeba_net import CelebaNet
from Models.logistic_regression_net import LinearClassificationNet
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from torch.utils.data import DataLoader


class ModelUtils:
    @staticmethod
    def create_private_model(
        model: torch.nn.Module,
        epsilon: float,
        original_optimizer,
        train_loader,
        epochs: int,
        delta: float,
        MAX_GRAD_NORM: float,
        batch_size: int,
        noise_multiplier: float = 0,
    ) -> Tuple[GradSampleModule, DPOptimizer, DataLoader]:
        """

        Args:
            model (torch.nn.Module): the model to wrap
            epsilon (float): the target epsilon for the privacy budget
            original_optimizer (_type_): the optimizer of the model before
                wrapping it with Privacy Engine
            train_loader (_type_): the train dataloader used to train the model
            epochs (_type_): for how many epochs the model will be trained
            delta (float): the delta for the privacy budget
            MAX_GRAD_NORM (float): the clipping value for the gradients
            batch_size (int): batch size

        Returns:
            Tuple[GradSampleModule, DPOptimizer, DataLoader]: the wrapped model,
                the wrapped optimizer and the train dataloader
        """
        privacy_engine = PrivacyEngine(accountant="rdp")

        # We can wrap the model with Privacy Engine using the
        # method .make_private(). This doesn't require you to
        # specify a epsilon. In this case we need to specify a
        # noise multiplier.
        # make_private_with_epsilon() instead requires you to
        # provide a target epsilon and a target delta. In this
        # case you don't need to specify a noise multiplier.
        if epsilon:
            (
                private_model,
                optimizer,
                train_loader,
            ) = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=original_optimizer,
                data_loader=train_loader,
                epochs=epochs,
                target_epsilon=epsilon,
                target_delta=delta,
                max_grad_norm=MAX_GRAD_NORM,
            )
        else:
            private_model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=original_optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=MAX_GRAD_NORM,
            )

        return private_model, optimizer, train_loader

    @staticmethod
    def get_model(
        dataset: str,
        device: torch.device,
        input_size: int = None,
        output_size: int = None,
    ) -> torch.nn.Module:
        """This function returns the model to train.

        Args:
            dataset (str): the name of the dataset
            device (torch.device): the device where the model will be trained

        Raises:
            ValueError: if the dataset is not supported

        Returns:
            torch.nn.Module: the model to train
        """
        if dataset == "celeba":
            return CelebaNet()
        elif dataset == "dutch":
            return LinearClassificationNet(input_size=11, output_size=2)
        elif dataset == "income":
            return LinearClassificationNet(input_size=54, output_size=2)

        else:
            raise ValueError(f"Dataset {dataset} not supported")
