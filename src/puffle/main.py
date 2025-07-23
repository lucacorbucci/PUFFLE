import argparse
import os
import random
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from opacus import PrivacyEngine
from torch import nn, optim

from puffle.PUFFLEModel.puffle_model import PUFFLEModel
from puffle.Regularization.disparity_loss import DisparityRegularizationLoss
from puffle.Regularization.mix_loss import MixLoss
from puffle.Utils.models import LinearClassificationNet
from puffle.Utils.utils import Utils

warnings.filterwarnings("ignore")


def setup_wandb(project_name: str, run_name: str | None):
    return wandb.init(project=project_name, name=run_name) if run_name else wandb.init(project=project_name)


def check_input(args):
    # check input for unfairness reduction parameters
    if args.unfairness_reduction:
        if args.regularization_lambda < 0 or args.regularization_lambda > 1:
            raise ValueError("Lambda must be between 0 and 1.")
        if args.fairness_metric not in ["disparity", "error_rate"]:
            raise ValueError("Fairness metric must be either 'disparity' or 'error_rate'.")
        if args.regularization_mode not in ["fixed", "tunable"]:
            raise ValueError("Regularization model must be either 'fixed' or 'tunable'.")
        if args.target is None:
            raise ValueError("Target must be specified for unfairness reduction.")
        if args.target < 0 or args.target > 1:
            raise ValueError("Target must be between 0 and 1.")


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

    print("CIAO")

    args = parser.parse_args()

    Utils.seed_everything()

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
    dutch_train, dutch_test, dutch_val = prepare_dutch("/raid/lcorbucci/data/dutch/", sweep=args.sweep)

    BATCH_SIZE = 256

    train_loader = torch.utils.data.DataLoader(
        dutch_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        dutch_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    if dutch_val is not None:
        val_loader = torch.utils.data.DataLoader(
            dutch_val,
            batch_size=BATCH_SIZE,
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
        poisson_sampling=True if private_training else False,
    )

    puffle_model = PUFFLEModel(
        model=model_gc,
        optimizer=optimizer_gc,
        criterion=criterion_gc,
        device=torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda"),
        lambda_regularization=args.regularization_lambda,
        wandb_run=wandb_run,
        target=args.target,
    )

    puffle_model.train(
        train_loader=train_loader_gc,
        epochs=epochs,
        val_loader=val_loader if dutch_val is not None else test_loader,
        verbose=True,
        average_probabilities=None,
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE,
    )

    if wandb_run:
        wandb_run.finish()
