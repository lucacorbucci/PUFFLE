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
    return wandb.init(project=project_name, name=run_name) if run_name else wandb.init(project=project_name)


def check_input(args):
    # check input for unfairness reduction parameters
    if args.unfairness_reduction:
        if args.regularization_lambda < 0 or args.regularization_lambda > 1:
            msg = "Lambda must be between 0 and 1."
            raise ValueError(msg)
        if args.fairness_metric not in ["disparity", "error_rate"]:
            msg = "Fairness metric must be either 'disparity' or 'error_rate'."
            raise ValueError(msg)
        if args.regularization_mode not in ["fixed", "tunable"]:
            msg = "Regularization model must be either 'fixed' or 'tunable'."
            raise ValueError(msg)
        if args.target is None:
            msg = "Target must be specified for unfairness reduction."
            raise ValueError(msg)
        if args.target < 0 or args.target > 1:
            msg = "Target must be between 0 and 1."
            raise ValueError(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Puffle with specified configuration.")

    # privacy parameters
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

    # Wandb parameters
    parser.add_argument("--wandb", type=bool, default=True)
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--run_name", type=int)
    parser.add_argument("--sweep", type=bool, default=False)

    # Unfairness reduction parameters
    parser.add_argument("--unfairness_reduction", type=float, default=False)
    parser.add_argument("--regularization_lambda", type=float, default=0)
    parser.add_argument("--fairness_metric", type=str, default="disparity")
    parser.add_argument("--regularization_mode", type=str, default="fixed")
    parser.add_argument("--target", type=float, default=None)
    parser.add_argument("--momentum", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--weight_decay_alpha", type=float, default=None)

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
        device=torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda"),
        lambda_regularization=args.regularization_lambda,
        wandb_run=wandb_run,
        target=args.target,
        tunable_lambda=args.regularization_mode == "tunable",
        momentum=args.momentum if args.momentum is not None else None,
        alpha=args.alpha if args.alpha is not None else None,
        weight_decay_alpha=args.weight_decay_alpha if args.weight_decay_alpha is not None else None,
    )

    puffle_model.train(
        train_loader=train_loader_gc,
        epochs=epochs,
        val_loader=val_loader if dutch_val is not None else None,
        test_loader=test_loader if dutch_test is not None else None,
        verbose=True,
        average_probabilities=None,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
    )

    if wandb_run:
        wandb_run.finish()
