import torch


class Utils:
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
