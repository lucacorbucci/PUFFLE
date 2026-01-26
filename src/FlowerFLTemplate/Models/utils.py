import torch

from Models.architectures import AbaloneNet, CelebaNet, LinearClassificationNet, SimpleMNISTModel


def get_model(
    dataset: str,
) -> torch.nn.Module:
    """
    Returns the appropriate PyTorch model instance based on the dataset name.

    Supports "celeba" (CelebaNet), "dutch"/"income"/"adult" (LinearClassificationNet with specific input/output sizes), "mnist" (SimpleMNISTModel), "abalone" (AbaloneNet with default hidden sizes).

    Args:
        dataset (str): Name of the dataset to select the model for.

    Returns:
        torch.nn.Module: Initialized model instance for the dataset.

    Raises:
        ValueError: If the dataset is not supported.
    """
    if dataset == "celeba":
        return CelebaNet()
    if dataset == "dutch":
        return LinearClassificationNet(input_size=12, output_size=2)
    if dataset == "income":
        return LinearClassificationNet(input_size=10, output_size=2)
    if dataset == "adult":
        return LinearClassificationNet(input_size=103, output_size=2)
    if dataset == "mnist":
        return SimpleMNISTModel()
    if dataset == "abalone":
        return AbaloneNet(input_size=8, hidden_sizes=[128, 64, 32], dropout_rate=0.2)

    raise ValueError(f"Dataset {dataset} not supported")
