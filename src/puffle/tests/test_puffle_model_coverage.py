from typing import cast
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import DataLoader

from puffle.PUFFLEModel.puffle_model import (
    PUFFLEModel,
    TrainingBatchResult,
)


class TestPUFFLEModelCoverage:
    def test_wandb_logging(self):
        model = torch.nn.Linear(2, 2)
        wandb_mock = MagicMock()
        puffle_model = PUFFLEModel(
            model=model,
            device="cpu",
            wandb_run=wandb_mock,
            tunable_lambda=True,
            target=0.1,
        )

        # Test _log_wandb_epoch
        metrics = {"loss": 0.5, "accuracy": 0.8, "f1": 0.7, "disparity": 0.2}
        puffle_model._log_wandb_epoch(metrics, epoch=0, mode="train")
        wandb_mock.log.assert_called_with(
            {
                "train_loss": 0.5,
                "train_accuracy": 0.8,
                "train_f1": 0.7,
                "train_disparity": 0.2,
                "epoch": 1,
            }
        )

        # Test validation mode with target penalty logic
        puffle_model._log_wandb_epoch(metrics, epoch=0, mode="val")
        # target=0.1. disparity=0.2. distance = 0.1 - 0.2 = -0.1.
        args, _ = wandb_mock.log.call_args
        assert "Custom_metric" in args[0]

    def test_update_lambda_wandb(self):
        model = torch.nn.Linear(2, 2)
        wandb_mock = MagicMock()
        puffle_model = PUFFLEModel(
            model=model, wandb_run=wandb_mock, tunable_lambda=True, target=0.1
        )
        puffle_model.update_lambda(unfairness_loss=0.5)
        assert puffle_model.lambda_regularization > 0

    def test_bmm_logic(self):
        # Trigger use_bmm logic
        optimizer = MagicMock()
        optimizer.signal_skip_step = MagicMock()
        model = MagicMock()

        puffle_model = PUFFLEModel(model=model, optimizer=optimizer)

        train_loader = MagicMock(spec=DataLoader)
        train_loader.batch_size = 32

        # Mock _run_training_loop to avoid actual loop
        with (
            patch.object(puffle_model, "_run_training_loop", return_value=({}, [])),
            patch("puffle.PUFFLEModel.puffle_model.BatchMemoryManager") as mock_bmm,
        ):
            puffle_model.train(train_loader, epochs=1, max_physical_batch_size=16)
            assert mock_bmm.called

    def test_update_alpha_logic(self):
        puffle_model = PUFFLEModel(model=MagicMock(), alpha=0.5, weight_decay_alpha=0.9)
        puffle_model.update_alpha(current_epoch=1)
        assert puffle_model.alpha == 0.45

        # Test exp_lr_scheduler static method
        val = PUFFLEModel.exp_lr_scheduler(1.0, 100, 0.01)
        assert val < 1.0

    def test_verbose_validation(self):
        model = MagicMock()
        puffle_model = PUFFLEModel(model=model)

        # Check stdout? Or just ensure it runs.
        metrics = {"train_loss": [0.5]}
        v_loader = [1]  # Truthy list
        with (
            patch.object(
                puffle_model,
                "evaluate",
                return_value={
                    "loss": 0.1,
                    "accuracy": 0.9,
                    "f1": 0.9,
                    "disparity": 0.0,
                },
            ) as mock_eval,
            patch.object(puffle_model, "_update_metrics_dict"),
            patch.object(puffle_model, "_log_wandb_epoch"),
        ):
            puffle_model._validate_and_test_epoch(
                epoch=0,
                epochs=1,
                metrics=metrics,
                val_loader=v_loader,
                test_loader=None,
                verbose=True,
            )
            assert mock_eval.called

    def test_train_one_epoch_wandb(self):
        # Test _train_one_epoch with wandb and tunable lambda to hit lines
        model = MagicMock()
        wandb_mock = MagicMock()
        puffle_model = PUFFLEModel(
            model=model, wandb_run=wandb_mock, tunable_lambda=True, alpha=0.1
        )

        # We need a dummy batch to iterate.
        # _train_one_epoch expects a DataLoader.
        train_loader = cast("DataLoader", [MagicMock()])

        with patch.object(puffle_model, "_train_batch") as mock_train:
            mock_train.return_value = TrainingBatchResult(
                loss=0.5,
                correct=1,
                total=1,
                y_batch=torch.tensor([0]),
                predicted=torch.tensor([0]),
                z_batch=torch.tensor([0]),
                unfairness=0.1,
            )
            puffle_model._train_one_epoch(train_loader, current_epoch=0)

        # Verify calls
        assert wandb_mock.log.called
