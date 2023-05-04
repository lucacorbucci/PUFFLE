import random

import torch
import torch.nn.functional as F
from torch import nn

random.seed(15)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


class RegularizationLoss(nn.Module):
    """This class defines the regularization loss as proposed in
    https://arxiv.org/abs/2302.09183.
    """

    def __init__(self, weight=None, size_average=True) -> None:
        """Initialization of the regularization loss."""
        super(RegularizationLoss, self).__init__()

    def forward(
        self,
        sensitive_attribute_list,
        targets,
        device,
        predictions,
        sensitive_attributes,
        possible_targets,
        temperature=0.01,
    ):
        """This function computes the regularization term.
        It takes as input the sensitive attribute list, the targets, the device and the predictions
        compute with the model. It returns the regularization term.

        What we do here:
        - We compute the log softmax of the predictions dividing by 0.01 as in the paper
        - Then we consider the possible combinations of targets and sensitive features
            and we compute the corresponding fairness violation term
        - Instead of returning the max of the fairness violation terms, we return the sum of all of them
            but, instead of directly summing them, we first multiply them by a mask. The mask is filled
            with 0.0001 for the fairness violation terms that are not the max and with 1 for the max.
            This is (at the moment) the only way to compute all the per sample gradients for the samples
            in the batch. If we use the torch.max instead of summing the fairness violation terms, we
            compute the per sample gradient only for some of the samples in the batch.


        Args:
            sensitive_attribute_list (_type_): a list with the value of the sensitive
                attribute for each sample in the batch
            targets (_type_): the expected target values for the samples in the batch
            device (_type_): the device we're using to train the model
            predictions (_type_): the output of the model
            sensitive_attributes (_type_): the possible values of the sensitive attribute
            possible_targets (_type_): the possible target values we have in this dataset

        Returns:
            float: the disparity metric computed on the data passed as parameter
        """

        fairness_violations = []
        softmax_ = F.log_softmax(predictions / temperature, dim=1)

        # predictions_argmax = torch.argmax(torch.tensor(predictions), dim=1).to(device)
        predictions_argmax = torch.argmax(predictions.clone().detach(), dim=1).to(
            device
        )

        for target in possible_targets:
            for z in sensitive_attributes:
                Z_eq_z = torch.sum(
                    torch.sum(softmax_[sensitive_attribute_list == z], dim=1)
                )

                Z_not_eq_z = torch.sum(
                    torch.sum(softmax_[sensitive_attribute_list != z], dim=1)
                )

                idx_true = (predictions_argmax == target) & (
                    sensitive_attribute_list == z
                )
                Y_eq_k_and_Z_eq_z = softmax_[idx_true]

                idx_true = (predictions_argmax == target) & (
                    sensitive_attribute_list != z
                )
                Y_eq_k_and_Z_not_eq_z = softmax_[idx_true]

                violation_term = torch.abs(
                    (
                        (torch.abs(torch.sum(torch.sum(Y_eq_k_and_Z_eq_z, dim=0))))
                        / (Z_eq_z + 0.0000001)
                    )
                    - (
                        (torch.abs(torch.sum(torch.sum(Y_eq_k_and_Z_not_eq_z, dim=0))))
                        / (Z_not_eq_z + 0.0000001)
                    )
                )

                fairness_violations.append(violation_term)

        fairness_violations_ = [item.item() for item in fairness_violations]
        index = fairness_violations_.index(max(fairness_violations_))
        fairness_violations = torch.stack(fairness_violations)

        mask = torch.full(
            (fairness_violations.shape[0],), 0.0001, dtype=torch.float32
        ).to(device)
        mask[index] = 1

        res = torch.sum(mask * fairness_violations)

        return res

    def violation_with_dataset(self, model, dataset, device):
        """
        When we want to compute the disparity metric on the entire dataset
        we can't directly use the forward function because we don't have the
        predictions and the sensitive attribute list for each batch.
        So in this function we just use the model to compute the predictions for
        all the samples in the dataset and we aggregate the results in a single
        final tensor that we pass to the forward function.

        Args:
            model (_type_): the model we want to evaluate
            dataset (_type_): the dataset we want to use during the evaluation
            device (_type_): the device we're using to train the model

        Returns:
            float: the disparity metric computed on the entire dataset
        """
        predictions = torch.tensor([]).to(device)
        sensitive_attribute_list = torch.tensor([]).to(device)
        targets = []
        model.eval()
        with torch.no_grad():
            for images, sensitive_attributes, target in dataset:
                images = images.to(device)
                target = target.to(device)

                output = model(images)

                predictions = torch.cat((predictions, output), 0)
                sensitive_attribute_list = torch.cat(
                    (sensitive_attribute_list, sensitive_attributes.to(device)), 0
                )
                # sensitive_attribute_list += sensitive_attributes.tolist()
                targets += target.tolist()

        fairness_violations = []
        sensitive_attributes = list(
            set([item.item() for item in sensitive_attribute_list])
        )
        target_list = list(set(targets))

        return self.forward(
            sensitive_attribute_list,
            targets,
            device,
            predictions,
            sensitive_attributes,
            target_list,
        )
