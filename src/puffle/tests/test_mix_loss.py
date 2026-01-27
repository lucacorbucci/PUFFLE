import torch
from torch import nn

from puffle.Regularization.mix_loss import MixLoss


class TestMixLoss:
    def test_mix_loss_initialization(self):
        model_loss = nn.CrossEntropyLoss()
        fair_loss = nn.Identity()  # Just for testing
        mix = MixLoss(model_loss, fair_loss, possible_sensitive_attributes=[0, 1, 2])

        assert mix.model_criterion is model_loss
        assert mix.unfairness_criterion is fair_loss
        assert mix.possible_sensitive_attributes == [0, 1, 2]

    def test_mix_loss_forward(self):
        model_loss = nn.CrossEntropyLoss()

        # Mock fairness loss that returns a constant
        class MockFairLoss(nn.Module):
            def forward(self, **_kwargs):
                return torch.tensor(1.0)

        mix = MixLoss(model_loss, MockFairLoss())

        # Input: (outputs, z_batch, lambda)
        outputs = torch.randn(4, 2)
        z_batch = torch.tensor([0, 1, 0, 1])
        lambda_val = 0.5
        target = torch.tensor([0, 1, 0, 1])

        loss = mix((outputs, z_batch, lambda_val), target)

        expected_ce = model_loss(outputs, target)
        expected_loss = 0.5 * expected_ce + 0.5 * 1.0

        assert torch.isclose(loss, expected_loss)
