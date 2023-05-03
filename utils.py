from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common.typing import Scalar
from Models.celeba import CelebaNet
from Models.cifar import CifarNet
from Models.mnist import MnistNet


class Utils:
    @staticmethod
    def get_model(dataset_name: str):
        if dataset_name == "cifar10":
            return CifarNet()
        elif dataset_name == "mnist":
            return MnistNet()
        elif dataset_name == "celeba":
            return CelebaNet()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    @staticmethod
    def get_params(model: torch.nn.ModuleList) -> List[np.ndarray]:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    @staticmethod
    def set_params(model: torch.nn.ModuleList, params: List[np.ndarray]):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        model.load_state_dict(state_dict, strict=True)
