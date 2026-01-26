# ignore warnings
import argparse
import os
import random
import warnings

import numpy as np
import torch
import wandb
from Models.logistic_regression_net import LinearClassificationNet
from utils.dutch import TabularDataset
from utils.model_utils import ModelUtils

from FairReg.DPLUtils.regularization_config import RegularizationConfig
from FairReg.Learning.learning_disparity import Learning

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


seed = 42
seed_everything(seed)

parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--target", type=float, default=0.5)
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("--regularization_lambda", type=float, default=0.1)

args = parser.parse_args()


wandb_run = wandb.init(
    # set the wandb project where this run will be logged
    project="Centralised Disparity",
    # track hyperparameters and run metadata
    config={
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "regularization_lambda": args.regularization_lambda,
    },
)

batch_size = args.batch_size

train_ds = torch.load("../dataset/celeba/train_ds.pt")
test_ds = torch.load("../dataset/celeba/test_ds.pt")

# sample 8000 samples from the train_ds.samples
indexes = np.random.choice(train_ds.samples.shape[0], 10000, replace=False)

samples = train_ds.samples[indexes]
targets = train_ds.targets[indexes]
sensitive_features = train_ds.sensitive_features[indexes]

validation_ds = TabularDataset(samples, sensitive_features, targets)

train_ds.samples = np.delete(train_ds.samples, indexes, axis=0)
train_ds.targets = np.delete(train_ds.targets, indexes, axis=0)
train_ds.sensitive_features = np.delete(train_ds.sensitive_features, indexes, axis=0)
train_ds.indexes = np.delete(train_ds.indexes, indexes, axis=0)

train_loader = torch.utils.data.DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

test_loader = torch.utils.data.DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

validation_loader = torch.utils.data.DataLoader(
    validation_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)

# Create the model that we will train, for Dutch we will use a LinearClassificationNet
# defined inside this Library in the Models/logistic_regression_net.py file
model = LinearClassificationNet()
model_regularization = LinearClassificationNet()

lr = args.lr

if args.optimizer == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer_regularization = torch.optim.SGD(model_regularization.parameters(), lr=lr)
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer_regularization = torch.optim.Adam(model_regularization.parameters(), lr=lr)

epochs = 10


# We don't want to use privacy in this case but we make the model private using
# the noise=0 because the code for learning the model only accept private models.
(
    private_model,
    private_optimizer,
    private_train_loader,
) = ModelUtils.create_private_model(
    model=model,
    epsilon=None,
    noise_multiplier=0,
    original_optimizer=optimizer,
    train_loader=train_loader,
    epochs=epochs,
    delta=0,
    MAX_GRAD_NORM=10000000000,  # since we just need to wrap the model without using privacy we use a high value here
    batch_size=batch_size,
)

# We don't want to use privacy in this case but we make the model private using
# the noise=0 because the code for learning the model only accept private models.

(
    private_model_regularization,
    private_optimizer_regularization,
    _,
) = ModelUtils.create_private_model(
    model=model_regularization,
    epsilon=None,
    noise_multiplier=0,
    original_optimizer=optimizer_regularization,
    train_loader=train_loader,
    epochs=epochs,
    delta=0,
    MAX_GRAD_NORM=10000000000,  # since we just need to wrap the model without using privacy we use a high value here
    batch_size=batch_size,
)


model.to("cuda")
private_model.to("cuda")

if args.target:
    train_parameters = RegularizationConfig(
        epochs=epochs,
        device="cuda",
        batch_size=batch_size,
        seed=seed,
        regularization=True,
        target=args.target,
        regularization_mode="fixed",
        regularization_lambda=args.regularization_lambda,
        optimizer="adam",
    )
else:
    train_parameters = RegularizationConfig(
        epochs=epochs,
        device="cuda",
        batch_size=batch_size,
        seed=seed,
        regularization=False,
        optimizer="adam",
    )


for epoch in range(0, epochs):
    # Now we can train the model. First of all we will train a model without any
    # fairness mitigation
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
        max_disparity_test,
        y_true,
        y_pred,
        sensitive_attributes,
        real_indexes,
        _,
    ) = Learning.test(
        model=private_model,
        test_loader=train_loader,
        train_parameters=train_parameters,
        current_epoch=epochs,
    )
    (
        _,
        accuracy_validation,
        _,
        _,
        _,
        max_disparity_test_validation,
        y_true_validation,
        y_pred_validation,
        sensitive_attributes_validation,
        _,
        _,
    ) = Learning.test(
        model=private_model,
        test_loader=validation_loader,
        train_parameters=train_parameters,
        current_epoch=epochs,
    )

    analysis_dict = {}
    for index, y, prediction, group in zip(real_indexes, y_true, y_pred, sensitive_attributes):
        if (y, prediction, group) not in analysis_dict:
            analysis_dict[(y, prediction, group)] = []
        analysis_dict[(y, prediction, group)].append(index)

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
            "Disparity Train": max_disparity_test,
            "Disparity Validation": max_disparity_test_validation,
            "Validation Accuracy": accuracy_validation,
            "Custom_metric": custom_metric,
            "Epoch": epoch,
        }
    )

wandb_run.finish()
