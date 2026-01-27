"""
Lambda Strategy Comparison Experiment

Compares three lambda update strategies (momentum, gradient, PID) under
distribution shift scenarios to evaluate adaptability and stability.
"""

import argparse
import time
import warnings

import torch
import wandb
from opacus import PrivacyEngine
from torch import nn, optim

from puffle.examples.data_preparation.dataset_preparation import prepare_dutch
from puffle.examples.models.models import LinearClassificationNet
from puffle.examples.utils.utils import seed_everything
from puffle.PUFFLEModel.puffle_model import PUFFLEModel
from puffle.Regularization.disparity_loss import DisparityRegularizationLoss
from puffle.Regularization.mix_loss import MixLoss

warnings.filterwarnings("ignore")


def setup_wandb(project_name: str, run_name: str | None):
    return (
        wandb.init(project=project_name, name=run_name)
        if run_name
        else wandb.init(project=project_name)
    )


def inject_distribution_shift(dataset, shift_type="increase_bias", shift_magnitude=0.3):
    """
    Inject distribution shift into dataset by modifying sensitive attributes.

    Args:
        dataset: PyTorch dataset
        shift_type: Type of shift ("increase_bias", "decrease_bias", "flip_labels")
        shift_magnitude: Magnitude of shift (0.0 to 1.0)

    Returns:
        Modified dataset

    """
    import numpy as np

    if shift_type == "increase_bias":
        # Increase correlation between sensitive attribute and label
        for i in range(len(dataset)):
            _x, z, y = dataset[i]
            if np.random.rand() < shift_magnitude:
                # Make z more predictive of y
                if y == 1 and z == 0:
                    dataset.sensitive_features[i] = 1
                elif y == 0 and z == 1:
                    dataset.sensitive_features[i] = 0

    elif shift_type == "decrease_bias":
        # Decrease correlation
        for i in range(len(dataset)):
            if np.random.rand() < shift_magnitude:
                dataset.sensitive_features[i] = 1 - dataset.sensitive_features[i]

    elif shift_type == "flip_labels":
        # Flip some labels to create sudden accuracy drop
        for i in range(len(dataset)):
            if np.random.rand() < shift_magnitude:
                dataset.targets[i] = 1 - dataset.targets[i]

    return dataset


def check_input(args):
    if args.regularization_lambda < 0 or args.regularization_lambda > 1:
        msg = "Lambda must be between 0 and 1."
        raise ValueError(msg)
    if args.target is None:
        msg = "Target must be specified."
        raise ValueError(msg)
    if args.target < 0 or args.target > 1:
        msg = "Target must be between 0 and 1."
        raise ValueError(msg)
    if args.lambda_update_strategy not in ["momentum", "gradient", "pid"]:
        msg = "Lambda update strategy must be 'momentum', 'gradient', or 'pid'."
        raise ValueError(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare lambda update strategies with distribution shift."
    )

    # Privacy parameters
    parser.add_argument("--epsilon", type=str, default=None)
    parser.add_argument("--noise_multiplier", type=float, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=10000000)

    # Training parameters
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_seed", type=int, default=None)
    parser.add_argument("--csv_path", type=str, default=None)

    # WandB parameters
    parser.add_argument("--wandb", type=bool, default=True)
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--sweep", type=bool, default=False)

    # Lambda tuning parameters
    parser.add_argument("--regularization_lambda", type=float, default=0.5)
    parser.add_argument("--target", type=float, required=True)
    parser.add_argument(
        "--lambda_update_strategy",
        type=str,
        required=True,
        choices=["momentum", "gradient", "pid"],
    )

    # Strategy-specific parameters
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--weight_decay_alpha", type=float, default=0.99)

    # PID parameters
    parser.add_argument("--lambda_kp", type=float, default=0.01)
    parser.add_argument("--lambda_ki", type=float, default=0.001)
    parser.add_argument("--lambda_kd", type=float, default=0.005)

    # Distribution shift parameters
    parser.add_argument(
        "--shift_epoch",
        type=int,
        default=None,
        help="Epoch at which to inject distribution shift (None = no shift)",
    )
    parser.add_argument(
        "--shift_type",
        type=str,
        default="increase_bias",
        choices=["increase_bias", "decrease_bias", "flip_labels"],
    )
    parser.add_argument(
        "--shift_magnitude",
        type=float,
        default=0.3,
        help="Magnitude of distribution shift (0.0 to 1.0)",
    )

    args = parser.parse_args()

    if args.validation_seed is None:
        validation_seed = int(str(time.time()).split(".")[1]) * args.seed
        args.validation_seed = validation_seed

    check_input(args)
    private_training = bool(args.noise_multiplier > 0 or args.epsilon is not None)

    wandb_run = (
        setup_wandb(
            project_name=args.project_name,
            run_name=args.run_name,
        )
        if args.wandb
        else None
    )

    # Log strategy and shift configuration
    if wandb_run:
        wandb_run.config.update(
            {
                "lambda_update_strategy": args.lambda_update_strategy,
                "shift_epoch": args.shift_epoch,
                "shift_type": args.shift_type if args.shift_epoch else "none",
                "shift_magnitude": args.shift_magnitude if args.shift_epoch else 0.0,
            }
        )

    seed_everything(args.seed)
    dutch_train, dutch_test, dutch_val = prepare_dutch(
        args.csv_path, sweep=args.sweep, validation_seed=args.validation_seed
    )
    seed_everything(args.seed)

    train_loader = torch.utils.data.DataLoader(
        dutch_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dutch_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    if dutch_val is not None:
        val_loader = torch.utils.data.DataLoader(
            dutch_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
    else:
        val_loader = None

    lr = args.lr
    epochs = args.epochs
    MAX_PHYSICAL_BATCH_SIZE = 1024

    privacy_engine = PrivacyEngine()
    criterion = MixLoss(
        model_loss=nn.CrossEntropyLoss(),
        unfairness_loss=DisparityRegularizationLoss(),
    )
    model = LinearClassificationNet(input_size=11, output_size=2)
    optimizer = (
        optim.SGD(model.parameters(), lr=lr, momentum=0)
        if args.optimizer == "sgd"
        else optim.Adam(model.parameters(), lr=lr)
    )

    model_gc, optimizer_gc, criterion_gc, train_loader_gc = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        criterion=criterion,
        grad_sample_mode="ghost",
        poisson_sampling=bool(private_training),
    )

    puffle_model = PUFFLEModel(
        model=model_gc,
        optimizer=optimizer_gc,
        criterion=criterion_gc,
        device=torch.device("cpu")
        if not torch.cuda.is_available()
        else torch.device("cuda"),
        lambda_regularization=args.regularization_lambda,
        wandb_run=wandb_run,
        target=args.target,
        tunable_lambda=True,
        lambda_update_strategy=args.lambda_update_strategy,
        momentum=args.momentum,
        alpha=args.alpha,
        weight_decay_alpha=args.weight_decay_alpha,
        lambda_kp=args.lambda_kp,
        lambda_ki=args.lambda_ki,
        lambda_kd=args.lambda_kd,
    )

    # Training loop with distribution shift injection
    for epoch in range(epochs):
        # Inject distribution shift at specified epoch
        if args.shift_epoch is not None and epoch == args.shift_epoch:
            print(f"\n{'=' * 60}")
            print(f"INJECTING DISTRIBUTION SHIFT AT EPOCH {epoch}")
            print(f"Type: {args.shift_type}, Magnitude: {args.shift_magnitude}")
            print(f"{'=' * 60}\n")

            dutch_train = inject_distribution_shift(
                dutch_train,
                shift_type=args.shift_type,
                shift_magnitude=args.shift_magnitude,
            )

            # Recreate data loader with shifted data
            train_loader = torch.utils.data.DataLoader(
                dutch_train,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

            _, _, _, train_loader_gc = privacy_engine.make_private(
                module=model_gc,
                optimizer=optimizer_gc,
                data_loader=train_loader,
                noise_multiplier=args.noise_multiplier,
                max_grad_norm=args.max_grad_norm,
                criterion=criterion_gc,
                grad_sample_mode="ghost",
                poisson_sampling=bool(private_training),
            )

            if wandb_run:
                wandb_run.log({"distribution_shift_injected": 1, "epoch": epoch})

        # Train for one epoch
        puffle_model.train(
            train_loader=train_loader_gc,
            epochs=1,
            val_loader=val_loader if dutch_val is not None else None,
            test_loader=test_loader if dutch_test is not None else None,
            verbose=True,
            average_probabilities=None,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        )

    if wandb_run:
        wandb_run.finish()
