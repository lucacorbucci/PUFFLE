import argparse
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
from opacus.validators import ModuleValidator

import wandb
from DPL.learning import Learning
from DPL.Utils.dataset_utils import DatasetUtils
from DPL.Utils.model_utils import ModelUtils
from DPL.Utils.train_parameters import TrainParameters

warnings.filterwarnings("ignore")


def get_device():
    """This function returns the device where the model will be trained."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def run_experiments(args):
    """Thi function runs the experiment."""
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    wandb_run = None
    if args.experiment_tag:
        tags = [item for item in args.experiment_tag.split(",")]
    if args.wandb:
        wandb_run = wandb.init(
            project="Experiment DPL",
            tags=[args.dataset] + tags if args.experiment_tag else [args.dataset],
            name=f"Centralised - Lambda {args.DPL_lambda}",
            config={
                "learning_rate": args.lr,
                "dataset": args.dataset,
                "csv": args.train_csv,
                "epochs": args.epochs,
                "private": args.private,
                "gradnorm": args.gradnorm,
                "epsilon": args.epsilon if args.private else 0,
                "delta": args.delta if args.private else 0,
                "DPL_regularization": args.DPL,
                "DPL_lambda": args.DPL_lambda,
                "batch_size": args.batch_size,
            },
        )

    # Download and prepare the dataset we will use for the experiment
    train_ds, test_ds = DatasetUtils.download_dataset(
        args.dataset, args.train_csv, args.test_csv, debug=args.debug
    )
    original_train_loader, test_loader = DatasetUtils.prepare_datasets(
        train_ds=train_ds,
        test_ds=test_ds,
        batch_size=args.batch_size,
    )

    device = f"cuda:{args.device}"
    # print(f"Training on cuda:{device}")

    # Get the model we want to train and the optimizer.
    # If we want to train a private model we have to
    # wrap it and call the crate_private_model function
    model = ModelUtils.get_model(args.dataset, device)
    model = ModuleValidator.fix(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    model_regularization = None
    optimizer_regularization = None

    if args.private:
        model, optimizer, train_loader = ModelUtils.create_private_model(
            model=model,
            epsilon=args.epsilon,
            original_optimizer=optimizer,
            train_loader=original_train_loader,
            epochs=args.epochs,
            delta=args.delta,
            MAX_GRAD_NORM=args.gradnorm,
            batch_size=args.batch_size,
        )
        if args.DPL:
            # If we want to use DPL with private training
            # we have to create a second model because
            # we can't just sum the two losses.
            model_regularization = ModelUtils.get_model(args.dataset, device)
            model_regularization = ModuleValidator.fix(model_regularization)
            optimizer_regularization = torch.optim.SGD(
                model_regularization.parameters(), lr=args.lr
            )
            (
                model_regularization,
                optimizer_regularization,
                _,
            ) = ModelUtils.create_private_model(
                model=model_regularization,
                epsilon=args.epsilon,
                original_optimizer=optimizer_regularization,
                train_loader=original_train_loader,
                epochs=args.epochs,
                delta=args.delta,
                MAX_GRAD_NORM=args.gradnorm,
                batch_size=args.batch_size,
            )
    else:
        _, _, train_loader = ModelUtils.create_private_model(
            model=model,
            epsilon=args.epsilon,
            original_optimizer=optimizer,
            train_loader=original_train_loader,
            epochs=args.epochs,
            delta=args.delta,
            MAX_GRAD_NORM=args.gradnorm,
            batch_size=args.batch_size,
        )

    model = model.to(device)
    if model_regularization:
        model_regularization.to(device)

    criterion = nn.CrossEntropyLoss()
    # print(args.DPL_lambda)

    # Create the TrainParameters object that will be used
    # to pass some of the training parameters to the
    # train_loop function
    train_parameters = TrainParameters(
        epochs=args.epochs,
        device=device,
        private=args.private,
        criterion=criterion,
        batch_size=args.batch_size,
        DPL=args.DPL,
        wandb_run=wandb_run,
        DPL_lambda=args.DPL_lambda,
        seed=args.seed,
        epsilon=args.epsilon,
    )

    # Actually train the model
    model = Learning.train_loop(
        train_parameters=train_parameters,
        model=model,
        model_regularization=model_regularization,
        optimizer=optimizer,
        optimizer_regularization=optimizer_regularization,
        train_loader=train_loader,
        test_loader=test_loader,
    )

    # In the end we save the model, if we want it and close wandb
    wandb.finish()
    if args.save_model:
        torch.save(
            model.state_dict(),
            f"saved_models/{args.dataset}_private_{args.private}_epsilon_{args.epsilon}_clipping_{args.gradnorm}_DPL_{args.DPL}_lambda_{args.DPL_lambda}.pt",
        )


def main():
    parser = argparse.ArgumentParser(description="Experiments with ColorMnist")
    parser.add_argument(
        "--private",
        type=bool,
        default=False,
        metavar="N",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        metavar="N",
        required=True,
    )
    parser.add_argument(
        "--train_csv",
        type=str,
        default="",
        metavar="N",
        required=True,
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="",
        metavar="N",
        required=True,
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default=None,
        metavar="N",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        metavar="N",
    )
    parser.add_argument(
        "--DPL_lambda",
        type=float,
        default=0,
        metavar="N",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0,
        metavar="N",
    )
    parser.add_argument(
        "--gradnorm",
        type=float,
        default=0,
        metavar="N",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        metavar="N",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        metavar="N",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0,
        metavar="N",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        metavar="N",
    )
    parser.add_argument(
        "--DPL",
        type=bool,
        default=False,
        metavar="N",
    )
    parser.add_argument(
        "--wandb",
        type=bool,
        default=False,
        metavar="N",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        metavar="N",
    )
    parser.add_argument(
        "--save_model",
        type=bool,
        default=False,
        metavar="N",
    )

    args = parser.parse_args()
    run_experiments(args)


if __name__ == "__main__":
    main()
