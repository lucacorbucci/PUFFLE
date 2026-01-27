from torch import nn


class MixLoss(nn.Module):
    def __init__(
        self,
        model_loss,
        unfairness_loss,
        *,
        possible_sensitive_attributes=None,
        possible_targets=None,
        reduction="mean",
        device="cpu",
    ):
        super().__init__()
        self.model_criterion = model_loss
        self.unfairness_criterion = unfairness_loss
        self.reduction = reduction
        self.possible_sensitive_attributes = possible_sensitive_attributes or [0, 1]
        self.possible_targets = possible_targets or [0, 1]
        self.device = device

    def forward(self, inputs, target):
        model_output = inputs[0]
        sensitive_value = inputs[1]
        lambda_regularization = inputs[2]

        model_loss = self.model_criterion(model_output, target)
        unfairness_loss = self.unfairness_criterion(
            sensitive_attribute_list=sensitive_value,
            device=self.device,
            predictions=model_output,
            possible_sensitive_attributes=self.possible_sensitive_attributes,
            possible_targets=self.possible_targets,
        )

        return (
            1 - lambda_regularization
        ) * model_loss + lambda_regularization * unfairness_loss
