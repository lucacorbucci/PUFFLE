from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common.typing import Scalar

from model_utils import ModelUtils
from Models.cifar import CifarNet
from Models.mnist import MnistNet


class Utils:
    @staticmethod
    def get_model(dataset_name: str):
        if dataset_name == "cifar10":
            return CifarNet()
        elif dataset_name == "mnist":
            return MnistNet()
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

    

    @staticmethod
    def get_evaluate_fn(
        testset,
        dataset_name: str,
    ) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
        """Return an evaluation function for centralized evaluation."""

        def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, float]]:
            """Use the entire CIFAR-10 test set for evaluation."""

            # determine device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # model = Net()
            model = Utils.get_model(dataset_name)
            Utils.set_params(model, parameters)
            model.to(device)

            testloader = torch.utils.data.DataLoader(testset, batch_size=50)
            loss, accuracy = ModelUtils.test(model, testloader, device=device)

            # return statistics
            return loss, {"accuracy": accuracy}

        return evaluate
