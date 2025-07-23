from torch import nn


class MixLoss(nn.Module):
    def __init__(self, model_loss, unfairness_loss, reduction="mean"):
        super(MixLoss, self).__init__()
        self.model_criterion = model_loss
        self.unfairness_criterion = unfairness_loss
        self.reduction = reduction

    def forward(self, input, target):
        model_output = input[0]
        sensitive_value = input[1]
        lambda_regularization = input[2]

        model_loss = self.model_criterion(model_output, target)
        unfairness_loss = self.unfairness_criterion(
            sensitive_attribute_list=sensitive_value,
            device="cpu",
            predictions=model_output,
            possible_sensitive_attributes=[0, 1],
            possible_targets=[0, 1],
        )
        return (1 - lambda_regularization) * model_loss + lambda_regularization * unfairness_loss
