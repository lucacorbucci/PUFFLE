"""
Lambda Strategy Comparison Experiment

Compares three lambda update strategies (momentum, gradient, PID) under
distribution shift scenarios using pre-generated datasets.
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
    parser.add_argument(
        "--epochs_before_shift",
        type=int,
        required=True,
        help="Number of epochs to train on original dataset",
    )
    parser.add_argument(
        "--epochs_after_shift",
        type=int,
        required=True,
        help="Number of epochs to train on shifted dataset",
    )
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--optimizer", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_seed", type=int, default=None)

    # Dataset paths
    parser.add_argument(
        "--csv_path_before",
        type=str,
        required=True,
        help="Path to directory containing dataset BEFORE shift",
    )
    parser.add_argument(
        "--csv_path_after",
        type=str,
        default=None,
        help="Path to directory containing dataset AFTER shift (optional)",
    )
    parser.add_argument(
        "--dataset_name_before",
        type=str,
        default="dutch_census_2001.csv",
        help="Filename of dataset before shift",
    )
    parser.add_argument(
        "--dataset_name_after",
        type=str,
        default=None,
        help="Filename of dataset after shift",
    )

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

    args = parser.parse_args()

    if args.validation_seed is None:
        validation_seed = int(str(time.time()).split(".")[1]) * args.seed
        args.validation_seed = validation_seed

    check_input(args)
    private_training = bool(args.noise_multiplier > 0 or args.epsilon is not None)

    # Determine if we have a distribution shift
    has_shift = args.csv_path_after is not None and args.dataset_name_after is not None
    total_epochs = args.epochs_before_shift + (
        args.epochs_after_shift if has_shift else 0
    )

    wandb_run = (
        setup_wandb(
            project_name=args.project_name,
            run_name=args.run_name,
        )
        if args.wandb
        else None
    )

    # Log configuration
    if wandb_run:
        wandb_run.config.update(
            {
                "lambda_update_strategy": args.lambda_update_strategy,
                "has_distribution_shift": has_shift,
                "shift_epoch": args.epochs_before_shift if has_shift else None,
                "dataset_before": args.dataset_name_before,
                "dataset_after": args.dataset_name_after if has_shift else "none",
                "total_epochs": total_epochs,
            }
        )

    # Phase 1: Train on original dataset
    print(f"\n{'=' * 60}")
    print("PHASE 1: Training on original dataset")
    print(f"Dataset: {args.dataset_name_before}")
    print(f"Epochs: {args.epochs_before_shift}")
    print(f"{'=' * 60}\n")

    seed_everything(args.seed)
    dutch_train, dutch_test, dutch_val = prepare_dutch(
        args.csv_path_before,
        sweep=args.sweep,
        validation_seed=args.validation_seed,
        dataset_name=args.dataset_name_before,
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

    # Train on original dataset
    puffle_model.train(
        train_loader=train_loader_gc,
        epochs=args.epochs_before_shift,
        val_loader=val_loader if dutch_val is not None else None,
        test_loader=test_loader if dutch_test is not None else None,
        verbose=True,
        average_probabilities=None,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
    )

    # Phase 2: Train on shifted dataset (if provided)
    if has_shift:
        print(f"\n{'=' * 60}")
        print("PHASE 2: DISTRIBUTION SHIFT - Switching to shifted dataset")
        print(f"Dataset: {args.dataset_name_after}")
        print(f"Epochs: {args.epochs_after_shift}")
        print(f"{'=' * 60}\n")

        if wandb_run:
            wandb_run.log(
                {
                    "distribution_shift_occurred": 1,
                    "epoch": args.epochs_before_shift,
                }
            )

        # Load shifted dataset
        seed_everything(args.seed)
        dutch_train_shifted, dutch_test_shifted, dutch_val_shifted = prepare_dutch(
            args.csv_path_after,
            sweep=args.sweep,
            validation_seed=args.validation_seed,
            dataset_name=args.dataset_name_after,
        )
        seed_everything(args.seed)

        # Create new data loaders
        train_loader_shifted = torch.utils.data.DataLoader(
            dutch_train_shifted,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        test_loader_shifted = torch.utils.data.DataLoader(
            dutch_test_shifted,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        if dutch_val_shifted is not None:
            val_loader_shifted = torch.utils.data.DataLoader(
                dutch_val_shifted,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )
        else:
            val_loader_shifted = None

        # Make shifted loader private
        _, _, _, train_loader_shifted_gc = privacy_engine.make_private(
            module=model_gc,
            optimizer=optimizer_gc,
            data_loader=train_loader_shifted,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            criterion=criterion_gc,
            grad_sample_mode="ghost",
            poisson_sampling=bool(private_training),
        )

        # Continue training on shifted dataset
        puffle_model.train(
            train_loader=train_loader_shifted_gc,
            epochs=args.epochs_after_shift,
            val_loader=val_loader_shifted if dutch_val_shifted is not None else None,
            test_loader=test_loader_shifted if dutch_test_shifted is not None else None,
            verbose=True,
            average_probabilities=None,
            max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
        )

    if wandb_run:
        wandb_run.finish()

    print(f"\n{'=' * 60}")
    print("Training complete!")
    print(f"Total epochs: {total_epochs}")
    if has_shift:
        print(f"Distribution shift at epoch: {args.epochs_before_shift}")
    print(f"{'=' * 60}\n")
