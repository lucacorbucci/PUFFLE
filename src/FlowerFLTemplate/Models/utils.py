from torch import nn

from FlowerFLTemplate.Models.architectures import (
    AbaloneNet,
    CelebaNet,
    LinearClassificationNet,
    SimpleMNISTModel,
)


def get_model(
    model_name: str,
    num_classes: int | None = None,
    in_channels: int | None = None,
    pixel: int | None = None,
) -> nn.Module:
    """
    Returns the model based on the model name.
    """
    if model_name == "LinearClassificationNet":
        if in_channels is None or num_classes is None:
            msg = "in_channels and num_classes must be provided for LinearClassificationNet"
            raise ValueError(msg)
        return LinearClassificationNet(input_size=in_channels, output_size=num_classes)
    if model_name == "AbaloneNet":
        if in_channels is None:
            msg = "in_channels must be provided for AbaloneNet"
            raise ValueError(msg)
        return AbaloneNet(input_size=in_channels)
    if model_name == "SimpleMNISTModel":
        # num_classes defaults to 10 in init, but allow override
        n_classes = num_classes if num_classes is not None else 10
        return SimpleMNISTModel(num_classes=n_classes)
    if model_name == "CelebaNet":
        n_classes = num_classes if num_classes is not None else 2
        in_c = in_channels if in_channels is not None else 3
        return CelebaNet(num_classes=n_classes, in_channels=in_c)
    msg = f"Unknown model name: {model_name}"
    raise ValueError(msg)
