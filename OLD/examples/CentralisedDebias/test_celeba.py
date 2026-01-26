import argparse
import os
import random
import warnings

import numpy as np
import torch
import wandb
from Models.celeba_net import CelebaNet

# from Models.logistic_regression_net import LinearClassificationNet
from torchvision import transforms
from utils.celeba import CelebaDataset
from utils.model_utils import ModelUtils

from FairReg.DPLUtils.regularization_config import RegularizationConfig
from FairReg.Learning.learning import Learning
from FairReg.Regularization.RegularizationLoss import RegularizationLoss

warnings.filterwarnings("ignore")


def seed_everything(seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--target", type=float, default=None)
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("--regularization_lambda", type=float, default=None)
parser.add_argument("--regularization_mode", type=str, default=None)
parser.add_argument("--alpha_target_lambda", type=float, default=None)
parser.add_argument("--momentum", type=float, default=None)
parser.add_argument("--weight_decay_lambda", type=float, default=None)

parser.add_argument("--sweep", type=bool, default=False)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--epsilon", type=float, default=None)
parser.add_argument("--clipping_value", type=float, default=100000000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="experiment")
parser.add_argument("--project_name", type=str, default="Centralised_Debias_Celeba")

args = parser.parse_args()


seed = args.seed
seed_everything(seed)

wandb_run = wandb.init(
    # set the wandb project where this run will be logged
    project=args.project_name,
    # set the name of the run
    name=args.run_name,
    # track hyperparameters and run metadata
    config={
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "regularization_lambda": args.regularization_lambda,
    },
)

batch_size = args.batch_size

transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ],
)
validation_loader = None
test_loader = None

if args.sweep:
    train_dataset = CelebaDataset(
        csv_path="../../../data/celeba/train.csv",
        image_path="../../../data/celeba/img_align_celeba",
        transform=transform,
        debug=True,
    )
    validation_dataset = CelebaDataset(
        csv_path="../../../data/celeba/validation.csv",
        image_path="../../../data/celeba/img_align_celeba",
        transform=transform,
        debug=True,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

else:
    train_dataset = CelebaDataset(
        csv_path="../../../data/celeba/train_with_validation.csv",
        image_path="../../../data/celeba/img_align_celeba",
        transform=transform,
        debug=True,
    )
    test_dataset = CelebaDataset(
        csv_path="../../../data/celeba/test.csv",
        image_path="../../../data/celeba/img_align_celeba",
        transform=transform,
        debug=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

seed_everything(42)
# Create the model that we will train, for Dutch we will use a LinearClassificationNet
# defined inside this Library in the Models/logistic_regression_net.py file
model = CelebaNet()
if args.target:
    model_regularization = CelebaNet()
else:
    model_regularization = None
lr = args.lr

if args.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    if args.target:
        optimizer_regularization = torch.optim.SGD(model_regularization.parameters(), lr=lr)
    else:
        optimizer_regularization = None
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if args.target:
        optimizer_regularization = torch.optim.Adam(model_regularization.parameters(), lr=lr)
    else:
        optimizer_regularization = None


# We don't want to use privacy in this case but we make the model private using
# the noise=0 because the code for learning the model only accept private models.
if args.epsilon:
    (
        private_model,
        private_optimizer,
        private_train_loader,
    ) = ModelUtils.create_private_model(
        model=model,
        epsilon=args.epsilon,
        original_optimizer=optimizer,
        train_loader=train_loader,
        epochs=args.epsilon,
        delta=1 / len(train_loader.dataset),
        MAX_GRAD_NORM=args.clipping_value,  # since we just need to wrap the model without using privacy we use a high value here
        batch_size=batch_size,
    )
else:
    (
        private_model,
        private_optimizer,
        private_train_loader,
    ) = ModelUtils.create_private_model(
        model=model,
        epsilon=None,
        original_optimizer=optimizer,
        train_loader=train_loader,
        epochs=args.epsilon,
        delta=0,
        MAX_GRAD_NORM=100000000000,  # since we just need to wrap the model without using privacy we use a high value here
        batch_size=batch_size,
        noise_multiplier=0,
    )

# We don't want to use privacy in this case but we make the model private using
# the noise=0 because the code for learning the model only accept private models.
if args.target:
    (
        private_model_regularization,
        private_optimizer_regularization,
        _,
    ) = ModelUtils.create_private_model(
        model=model_regularization,
        epsilon=args.epsilon,
        original_optimizer=optimizer_regularization,
        train_loader=train_loader,
        epochs=args.epsilon,
        delta=1 / len(train_loader.dataset),
        MAX_GRAD_NORM=args.clipping_value,  # since we just need to wrap the model without using privacy we use a high value here
        batch_size=batch_size,
    )
else:
    private_model_regularization = None
    private_optimizer_regularization = None


model.to("cuda")
private_model.to("cuda")

if args.target:
    if args.regularization_mode == "fixed":
        train_parameters = RegularizationConfig(
            epochs=args.epochs,
            device="cuda",
            batch_size=batch_size,
            seed=seed,
            regularization=True,
            target=args.target,
            regularization_mode=args.regularization_mode,
            regularization_lambda=args.regularization_lambda,
            optimizer="adam",
        )
    else:
        train_parameters = RegularizationConfig(
            epochs=args.epochs,
            device="cuda",
            batch_size=batch_size,
            seed=seed,
            regularization=True,
            target=args.target,
            regularization_mode=args.regularization_mode,
            regularization_lambda=args.regularization_lambda,
            optimizer="adam",
            momentum=args.momentum,
            alpha=args.alpha_target_lambda,
            weight_decay_alpha=args.weight_decay_lambda,
        )
else:
    train_parameters = RegularizationConfig(
        epochs=args.epochs,
        device="cuda",
        batch_size=batch_size,
        seed=seed,
        regularization=False,
        optimizer="adam",
    )


for epoch in range(0, args.epochs):
    # Now we can train the model. First of all we will train a model without any
    # fairness mitigation
    print(f"Epoch {epoch}")
    results = Learning.train_private_model(
        train_parameters=train_parameters,
        model=private_model,
        model_regularization=private_model_regularization,
        optimizer=private_optimizer,
        optimizer_regularization=private_optimizer_regularization,
        train_loader=train_loader,
        test_loader=test_loader,
        average_probabilities=None,
        current_epoch=epoch,
        wandb_run=wandb_run,
        epoch=epoch,
    )
    (
        _,
        accuracy,
        _,
        _,
        _,
        max_disparity_train,
        _,
        _,
        sensitive_attributes,
    ) = Learning.test(
        model=private_model,
        test_loader=train_loader,
        train_parameters=train_parameters,
        current_epoch=args.epochs,
    )
    if validation_loader:
        (
            _,
            accuracy_validation,
            _,
            _,
            _,
            max_disparity_test_validation,
            _,
            _,
            sensitive_attributes_validation,
        ) = Learning.test(
            model=private_model,
            test_loader=validation_loader,
            train_parameters=train_parameters,
            current_epoch=args.epochs,
        )

    if test_loader:
        (
            _,
            accuracy_validation,
            _,
            _,
            _,
            max_disparity_test_validation,
            _,
            _,
            sensitive_attributes_validation,
        ) = Learning.test(
            model=private_model,
            test_loader=test_loader,
            train_parameters=train_parameters,
            current_epoch=args.epochs,
        )

    custom_metric = accuracy_validation
    if args.target:
        w = 2
        distance = args.target - max_disparity_test_validation
        if distance > 0:
            penalty = 0
        else:
            penalty = -float("inf")  # w * distance
        custom_metric = accuracy_validation + penalty

    wandb_run.log(
        {
            "Train Accuracy": results["Train Accuracy"],
            "Train Loss": results["Train Loss"],
            "Train Loss + Regularizaion": results["Train Loss + Regularizaion"],
            "Validation Accuracy" if validation_loader else "Test Accuracy": accuracy_validation,
            "Custom_metric": custom_metric,
            "Epoch": epoch,
            "Disparity Train": max_disparity_train,
            "Disparity Validation" if validation_loader else "Disparity Test": max_disparity_test_validation,
        }
    )

wandb_run.finish()
