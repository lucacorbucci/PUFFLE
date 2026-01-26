# ignore warnings
import argparse
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import wandb
from Models.celeba_net import CelebaNetHair
from Models.logistic_regression_net import LinearClassificationNet
from scipy.io import arff
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from utils.dataset_utils import DatasetUtils
from utils.dutch import TabularDataset
from utils.model_utils import ModelUtils
from utils.tabular_datasets_utils import dataset_to_numpy, load_dutch

from FairReg.DPLUtils.regularization_config import RegularizationConfig
from FairReg.Learning.learning_new import Learning
from FairReg.Regularization.EqualizedOddsLoss import EqualizedOddsLoss
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


seed = 42
seed_everything(seed)


def compute_violation_with_argmax(
    sensitive_attribute_list: torch.tensor,
    analysis_dict: dict,
    y_pred: torch.tensor,
):
    return max(
        # FPR
        abs(
            (len(analysis_dict[(0, 1, 1)]) / (len(analysis_dict[(0, 1, 1)]) + len(analysis_dict[(0, 0, 1)])))
            - (len(analysis_dict[(0, 1, 0)]) / (len(analysis_dict[(0, 1, 0)]) + len(analysis_dict[(0, 0, 0)])))
        ),
        # TPR
        abs(
            (len(analysis_dict[(1, 1, 1)]) / (len(analysis_dict[(1, 1, 1)]) + len(analysis_dict[(1, 0, 1)])))
            - (len(analysis_dict[(1, 1, 0)]) / (len(analysis_dict[(1, 1, 0)]) + len(analysis_dict[(1, 0, 0)])))
        ),
    )


def compute_error_rate_difference(
    sensitive_attribute_list: torch.tensor,
    analysis_dict: dict,
    y_pred: torch.tensor,
):
    return abs(
        (
            (len(analysis_dict[(0, 1, 0)]) + len(analysis_dict[(1, 0, 0)]))
            / (
                len(analysis_dict[(0, 1, 0)])
                + len(analysis_dict[(1, 0, 0)])
                + len(analysis_dict[(1, 1, 0)])
                + len(analysis_dict[(0, 0, 0)])
            )
        )
        - (
            (len(analysis_dict[(0, 1, 1)]) + len(analysis_dict[(1, 0, 1)]))
            / (
                len(analysis_dict[(0, 1, 1)])
                + len(analysis_dict[(1, 0, 1)])
                + len(analysis_dict[(1, 1, 1)])
                + len(analysis_dict[(0, 0, 1)])
            )
        )
    )


parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--target", type=float, default=0.5)
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("--regularization_lambda", type=float, default=0.1)
parser.add_argument("--regularization_mode", type=str, default="fixed")
parser.add_argument("--alpha_target_lambda", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.1)
parser.add_argument("--weight_decay_lambda", type=float, default=0.1)


args = parser.parse_args()


wandb_run = wandb.init(
    # set the wandb project where this run will be logged
    project="Equalised_odds",
    # track hyperparameters and run metadata
    config={
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "optimizer": args.optimizer,
        "regularization_lambda": args.regularization_lambda,
    },
)

batch_size = args.batch_size

# torch.save(train_ds, "../dataset/dutch/train_ds.pt")
# torch.save(test_ds, "../dataset/dutch/test_ds.pt")

train_ds = torch.load("../dataset/dutch/train_ds.pt")
test_ds = torch.load("../dataset/dutch/test_ds.pt")

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
model = CelebaNetHair()
model_regularization = CelebaNetHair()

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
    if args.regularization_mode == "fixed":
        train_parameters = RegularizationConfig(
            epochs=epochs,
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
            epochs=epochs,
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

    # my_eq_odds = compute_violation_with_argmax(sensitive_attribute_list=np.array(sensitive_attributes), analysis_dict=analysis_dict, y_pred=np.array(y_pred))
    # my_error_rate_difference = compute_error_rate_difference(sensitive_attribute_list=np.array(sensitive_attributes), analysis_dict=analysis_dict, y_pred=np.array(y_pred))

    disparity_train = RegularizationLoss().compute_violation_with_argmax(
        predictions_argmax=y_pred,
        sensitive_attribute_list=sensitive_attributes,
        current_target=0,
        current_sensitive_feature=0,
    )
    disparity_validation = RegularizationLoss().compute_violation_with_argmax(
        predictions_argmax=y_pred_validation,
        sensitive_attribute_list=sensitive_attributes_validation,
        current_target=0,
        current_sensitive_feature=0,
    )
    print(
        f"Epoch {epoch} - Train accuracy {results['Train Accuracy']} - Train Loss {results['Train Loss']} - Equalized Odds Difference {max_disparity_test} - Regularization Loss {results['Train Loss + Regularizaion']}"
    )

    print(
        f"Epoch {epoch} - Validation accuracy {accuracy_validation} - Equalized Odds Difference Validation {max_disparity_test_validation}"
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
            "Equalized Odds Difference Train": max_disparity_test,
            "Equalized Odds Difference Validation": max_disparity_test_validation,
            "Validation Accuracy": accuracy_validation,
            # "My Equalized Odds Difference": my_eq_odds,
            "Custom_metric": custom_metric,
            # "Error Rate Difference": my_error_rate_difference,
            "Epoch": epoch,
            "Disparity Train": disparity_train,
            "Disparity Validation": disparity_validation,
        }
    )

wandb_run.finish()
