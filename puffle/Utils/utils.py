import json
import os
import random
import shutil
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from FederatedDataset.PartitionTypes.iid_partition import IIDPartition
from FederatedDataset.PartitionTypes.non_iid_partition_with_sensitive_feature import (
    NonIIDPartitionWithSensitiveFeature,
)
from FederatedDataset.PartitionTypes.representative import Representative
from FederatedDataset.Utils.utils import PartitionUtils
from PIL import Image
from Utils.model_utils import ModelUtils
from Utils.train_parameters import TrainParameters
from flwr.common.typing import Scalar
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset

from DPL.Learning.learning import Learning
from DPL.Regularization.RegularizationLoss import RegularizationLoss


class Utils:
    # plot the bar plot of the disparities
    @staticmethod
    def plot_bar_plot(title: str, disparities: list, nodes: list):
        plt.figure(figsize=(20, 8))
        plt.bar(range(len(disparities)), disparities)
        plt.xticks(range(len(nodes)), nodes)
        plt.title(title)
        # add a vertical line on xtick=75
        plt.axvline(x=75, color="r", linestyle="--")
        plt.xticks(rotation=90)
        # plt.show()
        # font size x axis
        plt.rcParams.update({"font.size": 10})
        plt.savefig(f"./{title}.png")
        plt.tight_layout()

    @staticmethod
    def get_noise(
        mechanism_type: str,
        epsilon: float = None,
        sensitivity: float = None,
        sigma: float = None,
    ):
        if mechanism_type == "laplace":
            return np.random.laplace(loc=0, scale=sensitivity / epsilon, size=1)
        elif mechanism_type == "geometric":
            p = 1 - np.exp(-epsilon / sensitivity)
            return (
                np.random.geometric(p=p, size=1) - np.random.geometric(p=p, size=1)
            )[0]
        elif mechanism_type == "gaussian":
            return np.random.normal(loc=0, scale=sigma, size=1)[0]
        else:
            raise ValueError(
                "The mechanism type must be either laplace, geometric or gaussian"
            )

    @staticmethod
    def seed_everything(seed: int):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

    @staticmethod
    def setup_wandb(args, train_parameters):
        private = False if args.epsilon is None else True

        if not args.sweep:
            name = "experiment" if not args.run_name else args.run_name
            wandb_run = wandb.init(
                # set the wandb project where this run will be logged
                project=(
                    "FL_fairness" if args.project_name is None else args.project_name
                ),
                name=name,
                # track hyperparameters and run metadata
                config={
                    "learning_rate": args.lr,
                    "csv": args.train_csv,
                    "DPL_regularization": args.regularization,
                    # "DPL_lambda": args.DPL_lambda,
                    "batch_size": args.batch_size,
                    "dataset": args.dataset,
                    "num_rounds": args.num_rounds,
                    "pool_size": args.pool_size,
                    "sampled_clients": args.sampled_clients,
                    "epochs": args.epochs,
                    "private": private,
                    "epsilon": args.epsilon,
                    "gradnorm": args.clipping,
                    "probability_estimation": args.probability_estimation,
                    "perfect_probability_estimation": args.perfect_probability_estimation,
                    "alpha": args.alpha,
                    "percentage_unbalanced_nodes": args.percentage_unbalanced_nodes,
                    "alpha_target_lambda": args.alpha_target_lambda,
                    "target": args.target,
                    "weight_decay_lambda": args.weight_decay_lambda,
                    "regularization_mode": args.regularization_mode,
                    "regularization_lambda": args.regularization_lambda,
                    "momentum": args.momentum,
                    "node_shuffle_seed": args.node_shuffle_seed,
                    "unbalanced_ratio": args.unbalanced_ratio,
                },
            )
        else:
            wandb_run = wandb.init(
                # set the wandb project where this run will be logged
                project=(
                    "FL_fairness" if args.project_name is None else args.project_name
                ),
                # name=f"FL - Lambda {args.DPL_lambda} - LR {args.lr} - Batch {args.batch_size}",
                # track hyperparameters and run metadata
                config={
                    "learning_rate": args.lr,
                    "csv": args.train_csv,
                    "DPL_regularization": args.regularization,
                    # "DPL_lambda": args.DPL_lambda,
                    "batch_size": args.batch_size,
                    "dataset": args.dataset,
                    "num_rounds": args.num_rounds,
                    "pool_size": args.pool_size,
                    "sampled_clients": args.sampled_clients,
                    "epochs": args.epochs,
                    "private": private,
                    "epsilon": args.epsilon,
                    "gradnorm": args.clipping,
                    # "delta": args.delta if args.private else 0,
                    # "noise_multiplier": noise_multiplier,
                    "probability_estimation": args.probability_estimation,
                    "perfect_probability_estimation": args.perfect_probability_estimation,
                    "alpha": args.alpha,
                    "percentage_unbalanced_nodes": args.percentage_unbalanced_nodes,
                    "alpha_target_lambda": args.alpha_target_lambda,
                    "target": args.target,
                    "weight_decay_lambda": args.weight_decay_lambda,
                    "regularization_mode": args.regularization_mode,
                    "regularization_lambda": args.regularization_lambda,
                    "momentum": args.momentum,
                    "node_shuffle_seed": args.node_shuffle_seed,
                    "unbalanced_ratio": args.unbalanced_ratio,
                },
            )
        return wandb_run

    @staticmethod
    def get_dataset_statistics(client_dataset, client_disparity, client_metadata):
        sens_features = client_dataset.sensitive_features
        targets = client_dataset.targets
        sens_features_and_targets = list(zip(targets, sens_features))
        counter_combination = Counter(sens_features_and_targets)
        counter_sens_features = Counter(sens_features)
        counter_targets = Counter(targets)

        # Return a dictionary with the statistics of the dataset
        return {
            "counter_combination": {
                str(key): value for key, value in counter_combination.items()
            },
            "counter_sens_features": {
                str(key): value for key, value in counter_sens_features.items()
            },
            "counter_targets": {
                str(key): value for key, value in counter_targets.items()
            },
            "client_disparity": client_disparity,
            "unfair_client": client_metadata,
        }

    # DEBUG
    def compute_disparities_debug(nodes):
        possible_clients = []
        for client in nodes:
            possible_z = np.array([])
            possible_y = np.array([])
            tmp_y = []
            tmp_z = []
            for sample in client:
                tmp_y.append(sample["y"])
                tmp_z.append(sample["z"])

            unique_z = np.unique(np.array(tmp_z))
            unique_y = np.unique(np.array(tmp_y))
            possible_z = np.unique(np.concatenate((possible_z, unique_z)))
            possible_y = np.unique(np.concatenate((possible_y, unique_y)))
            possible_clients.append((possible_y, possible_z))

        disparities = []
        for node, possible_client in zip(nodes, possible_clients):
            possible_z = possible_client[1]
            possible_y = possible_client[0]
            max_disparity = np.max(
                [
                    RegularizationLoss().compute_violation_with_argmax(
                        predictions_argmax=(
                            np.array([sample["y"] for sample in node])
                            if isinstance(node, list)
                            else np.array(node["y"])
                        ),
                        sensitive_attribute_list=(
                            np.array([sample["z"] for sample in node])
                            if isinstance(node, list)
                            else np.array(node["z"])
                        ),
                        current_target=int(target),
                        current_sensitive_feature=int(sv),
                    )
                    for target in possible_y
                    for sv in possible_z
                ]
            )
            disparities.append(max_disparity)
        print(f"Mean of disparity {np.mean(disparities)} - std {np.std(disparities)}")
        return disparities

    @staticmethod
    def get_dataset_statistics_with_lists(nodes, client_disparity, client_metadata):
        dictionaries = []
        for node, disparity, metadata in zip(nodes, client_disparity, client_metadata):
            sens_features = [item.item() for item in node["z"]]
            targets = [item.item() for item in node["y"]]
            sens_features_and_targets = list(zip(targets, sens_features))

            counter_combination = Counter(sens_features_and_targets)
            counter_sens_features = Counter(sens_features)
            counter_targets = Counter(targets)

            dictionary = {
                "counter_combination": {
                    str(key): value for key, value in counter_combination.items()
                },
                "counter_sens_features": {
                    str(key): value for key, value in counter_sens_features.items()
                },
                "counter_targets": {
                    str(key): value for key, value in counter_targets.items()
                },
                "client_disparity": disparity,
                "unfair_client": metadata,
            }
            dictionaries.append(dictionary)
        return dictionaries

    @staticmethod
    def get_optimizer(model, train_parameters, lr):
        if train_parameters.optimizer == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=lr,
            )
        elif train_parameters.optimizer == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=lr,
            )
        elif train_parameters.optimizer == "adamW":
            return torch.optim.AdamW(
                model.parameters(),
                lr=lr,
            )
        else:
            raise ValueError("Optimizer not recognized")

    @staticmethod
    def rescale_lambda(value, old_min, old_max, new_min, new_max):
        old_range = old_max - old_min
        new_range = new_max - new_min
        return (((value - old_min) * new_range) / old_range) + new_min

    @staticmethod
    def get_params(model: torch.nn.ModuleList) -> List[np.ndarray]:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    @staticmethod
    def set_params(model: torch.nn.ModuleList, params: List[np.ndarray]):
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        model.load_state_dict(state_dict, strict=True)

    @staticmethod
    def get_transformation(dataset_name: str):
        if dataset_name == "cifar10":
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ],
            )
        elif dataset_name == "mnist":
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ],
            )
        elif dataset_name == "celeba":
            return transforms.Compose(
                [
                    transforms.Resize((64, 64)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ],
            )
        else:
            return None

    @staticmethod
    def get_dataset(path_to_data: Path, cid: str, partition: str, dataset: str):
        # generate path to cid's data
        path_to_data = path_to_data / cid / (partition + ".pt")
        if dataset == "dutch":
            return torch.load(path_to_data)
        elif dataset == "income":
            return torch.load(path_to_data)
        else:
            return TorchVision_FL(
                path_to_data,
                transform=Utils.get_transformation(dataset),
            )

    @staticmethod
    def get_random_id_splits(total: int, val_ratio: float, shuffle: bool = True):
        """splits a list of length `total` into two following a
        (1-val_ratio):val_ratio partitioning.

        By default the indices are shuffled before creating the split and
        returning.
        """

        if isinstance(total, int):
            indices = list(range(total))
        else:
            indices = total

        split = int(np.floor(val_ratio * len(indices)))
        # print(f"Users left out for validation (ratio={val_ratio}) = {split} ")
        if shuffle:
            np.random.shuffle(indices)
        return indices[split:], indices[:split]

    @staticmethod
    def do_fl_partitioning(
        path_to_dataset: str,
        pool_size: int,
        num_classes: int,
        partition_type: str,
        val_ratio: float = 0.0,
        alpha: float = 1,
        train_parameters: TrainParameters = None,
        partition: str = "train",
        group_to_reduce=None,
        group_to_increment=None,
        number_of_samples_per_node=None,
        ratio_unfair_nodes=None,
        ratio_unfairness=None,
        one_group_nodes: bool = False,
        splitted_data_dir: str = None,
    ):
        print("Partitioning the dataset")

        images, sensitive_attribute, labels = torch.load(path_to_dataset)
        mapping = {-1: 0, 1: 1, 0: 0}
        if (
            train_parameters.metric == "disparity"
            or train_parameters.metric == "equalised_odds"
        ):
            sensitive_attribute = torch.tensor(
                [
                    mapping[item] if item in mapping else item
                    for item in sensitive_attribute
                ]
            )
        else:
            sensitive_attribute = torch.tensor([item for item in sensitive_attribute])

        idx = torch.tensor(list(range(len(images))))
        dataset = [idx, sensitive_attribute, labels]
        print(Counter(labels))

        metadata = [0] * pool_size

        if partition_type == "iid":
            splitted_indexes = IIDPartition.do_iid_partitioning_with_indexes(
                indexes=idx,
                num_partitions=pool_size,
            )
            partitions = PartitionUtils.create_splitted_dataset_from_tuple(
                splitted_indexes=splitted_indexes,
                dataset=dataset,
            )
        elif partition_type == "non_iid":
            (
                splitted_indexes,
                _,
                partitions_index_list,
                _,
            ) = NonIIDPartitionWithSensitiveFeature.do_partitioning_with_dataset_list(
                labels=labels,
                sensitive_features=sensitive_attribute,
                num_partitions=pool_size,
                alpha=alpha,
                total_num_classes=2,
            )
            partitions = PartitionUtils.create_splitted_dataset_from_tuple(
                splitted_indexes=partitions_index_list,
                dataset=dataset,
            )

        elif partition_type == "representative":
            print("SPLITTING THE DATASET USING the representative partitioning")

            partitions_index_list, metadata = Representative.do_partitioning(
                labels=labels,
                sensitive_features=sensitive_attribute,
                num_partitions=pool_size,
                total_num_classes=2,
                group_to_reduce=group_to_reduce,
                group_to_increment=group_to_increment,
                number_of_samples_per_node=number_of_samples_per_node,
                ratio_unfair_nodes=ratio_unfair_nodes,
                ratio_unfairness=ratio_unfairness,
                one_group_nodes=one_group_nodes,
            )
            partitions = PartitionUtils.create_splitted_dataset_from_tuple(
                splitted_indexes=partitions_index_list,
                dataset=dataset,
            )
        else:
            raise ValueError(f"Unknown partition type: {partition_type}")

        # now save partitioned dataset to disk
        # first delete dir containing splits (if exists), then create it
        splits_dir = path_to_dataset.parent / splitted_data_dir
        if splits_dir.exists() and partition == "train":
            shutil.rmtree(splits_dir)
            Path.mkdir(splits_dir, parents=True)

        nodes = []
        datasets = []
        possible_z = np.array([])
        possible_y = np.array([])
        for p in range(pool_size):
            labels = partitions[p][2]
            sensitive_features = partitions[p][1]

            image_idx = partitions[p][0]
            imgs = [images[image_id] for image_id in image_idx]

            # create dir
            if not (splits_dir / str(p)).exists():
                Path.mkdir(splits_dir / str(p))

            nodes.append({"y": labels, "z": sensitive_features})
            unique_z = np.unique(np.array(sensitive_features))
            unique_y = np.unique(np.array(labels))
            possible_z = np.unique(np.concatenate((possible_z, unique_z)))
            possible_y = np.unique(np.concatenate((possible_y, unique_y)))
            with open(
                splits_dir
                / str(p)
                / ("train.pt" if partition == "train" else "test.pt"),
                "wb",
            ) as f:
                torch.save([imgs, sensitive_features, labels], f)

        # convert nodes into the format expected by compute_disparities_debug
        tmp_nodes = []
        predictions = []
        sensitive_features = []
        for node in nodes:
            predictions.append([int(item) for item in node["y"]])
            sensitive_features.append([int(item) for item in node["z"]])
            current_node = []
            for y, z in zip(node["y"], node["z"]):
                current_node.append({"y": int(y), "z": int(z)})
            if current_node:
                tmp_nodes.append(current_node)
        disparities = Utils.compute_disparities_debug(tmp_nodes)
        Utils.plot_bar_plot(
            title="Distribution Disparities",
            disparities=disparities,
            nodes=[f"{i}" for i in range(len(nodes))],
        )
        possible_y = [str(int(item)) for item in possible_y.tolist()]
        possible_z = [str(int(item)) for item in possible_z.tolist()]
        # we are still assuming a binary target
        # however, we can have a non binary sensitive value
        missing_combinations = []
        all_combinations = []
        sent_disparity_combinations = [f"1|{sensitive}" for sensitive in possible_z]
        for combination in sent_disparity_combinations:
            missing_combinations.append(("0" + combination[1:], combination))
            all_combinations.append(combination)
            all_combinations.append("0" + combination[1:])

        json_file = {
            "possible_z": possible_z,
            "possible_y": possible_y,
            "missing_combinations": missing_combinations,
            "all_combinations": all_combinations,
            "combinations": sent_disparity_combinations,
        }
        with open(f"{splits_dir}/metadata.json", "w") as outfile:
            json_object = json.dumps(json_file, indent=4)
            outfile.write(json_object)

        counter_distribution_nodes = Utils.compute_distribution_debug(
            predictions=predictions, sensitive_features=sensitive_features
        )
        Utils.plot_distributions(
            title="Distribution of the nodes",
            counter_groups=counter_distribution_nodes,
            nodes=[f"{i}" for i in range(len(counter_distribution_nodes))],
            all_combinations=all_combinations,
        )

        return splits_dir

    @staticmethod
    def plot_distributions(
        title: str, counter_groups: list, nodes: list, all_combinations: list
    ):
        plt.figure(figsize=(20, 8))
        previous_sum = []
        for combination in all_combinations:
            counter = [
                counter[(int(combination[0]), int(combination[-1]))]
                for counter in counter_groups
            ]
            print(counter)
            if previous_sum:
                plt.bar(range(len(counter)), counter, bottom=previous_sum)
            else:
                plt.bar(range(len(counter)), counter)
                previous_sum = [0 for _ in counter]

            previous_sum = [sum(x) for x in zip(previous_sum, counter)]

        # counter_group_0_0 = [counter[(0, 0)] for counter in counter_groups]
        # counter_group_0_1 = [counter[(0, 1)] for counter in counter_groups]
        # counter_group_1_0 = [counter[(1, 0)] for counter in counter_groups]
        # counter_group_1_1 = [counter[(1, 1)] for counter in counter_groups]

        # plot a barplot with counter_group_0_0, counter_group_0_1, counter_group_1_0, counter_group_1_1
        # for each client in the same plot

        # plt.bar(range(len(counter_group_0_0)), counter_group_0_0)
        # plt.bar(range(len(counter_group_0_1)), counter_group_0_1, bottom=counter_group_0_0)
        # plt.bar(
        #     range(len(counter_group_1_0)),
        #     counter_group_1_0,
        #     bottom=[sum(x) for x in zip(counter_group_0_0, counter_group_0_1)],
        # )
        # plt.bar(
        #     range(len(counter_group_1_1)),
        #     counter_group_1_1,
        #     bottom=[
        #         sum(x) for x in zip(counter_group_0_0, counter_group_0_1, counter_group_1_0)
        #     ],
        # )

        plt.xlabel("Client")
        plt.ylabel("Amount of samples")
        plt.title("Samples for each group (target/sensitive Value) per client")
        plt.legend(all_combinations)
        # font size 20
        plt.rcParams.update({"font.size": 20})
        plt.rcParams.update({"font.size": 10})
        plt.savefig(f"./{title}.png")
        plt.tight_layout()

    @staticmethod
    def compute_distribution_debug(predictions, sensitive_features):
        counter_nodes = []
        for prediction, sensitive_feature in zip(predictions, sensitive_features):
            counter_node = []
            for pred, sf in zip(prediction, sensitive_feature):
                counter_node.append((pred, sf))
            counter_nodes.append(Counter(counter_node))
        return counter_nodes

    @staticmethod
    def prepare_dataset_for_FL(
        dataset,
        base_path: str,
        dataset_name: str,
        partition: str = "train",
    ):
        # fuse all data splits into a single "training.pt"
        data_loc = Path(base_path) / f"{dataset_name}/{dataset_name}-10-batches-py"
        train_path = data_loc / ("training.pt" if partition == "train" else "test.pt")
        print("Generating unified dataset")
        torch.save(
            [
                dataset.samples,
                np.array(dataset.sensitive_attributes),
                np.array(dataset.targets),
            ],
            train_path,
        )

        print("Data Correctly Loaded")

        return train_path

    @staticmethod
    def get_dataloader(
        path_to_data: str,
        cid: str,
        # is_train: bool,
        batch_size: int,
        workers: int,
        dataset: str,
        partition: str = "train",
    ):
        """Generates trainset/valset object and returns appropiate dataloader."""

        partition = (
            "train"
            if partition == "train"
            else "test" if partition == "test" else "val"
        )
        dataset = Utils.get_dataset(Path(path_to_data), cid, partition, dataset)

        # we use as number of workers all the cpu cores assigned to this actor
        kwargs = {"num_workers": workers, "pin_memory": True, "drop_last": False}
        return DataLoader(dataset, batch_size=batch_size, **kwargs)

    @staticmethod
    def create_private_model(
        model: torch.nn.Module,
        epsilon: float,
        original_optimizer,
        train_loader,
        epochs: int,
        delta: float,
        MAX_GRAD_NORM: float,
        batch_size: int,
        noise_multiplier: float = 0,
        accountant=None,
    ) -> Tuple[GradSampleModule, DPOptimizer, DataLoader]:
        """

        Args:
            model (torch.nn.Module): the model to wrap
            epsilon (float): the target epsilon for the privacy budget
            original_optimizer (_type_): the optimizer of the model before
                wrapping it with Privacy Engine
            train_loader (_type_): the train dataloader used to train the model
            epochs (_type_): for how many epochs the model will be trained
            delta (float): the delta for the privacy budget
            MAX_GRAD_NORM (float): the clipping value for the gradients
            batch_size (int): batch size

        Returns:
            Tuple[GradSampleModule, DPOptimizer, DataLoader]: the wrapped model,
                the wrapped optimizer and the train dataloader
        """
        privacy_engine = PrivacyEngine(accountant="rdp")
        if accountant:
            privacy_engine.accountant = accountant

        # We can wrap the model with Privacy Engine using the
        # method .make_private(). This doesn't require you to
        # specify a epsilon. In this case we need to specify a
        # noise multiplier.
        # make_private_with_epsilon() instead requires you to
        # provide a target epsilon and a target delta. In this
        # case you don't need to specify a noise multiplier.
        if epsilon:
            print(f"Creating private model using epsilon {epsilon}")
            (
                private_model,
                optimizer,
                train_loader,
            ) = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=original_optimizer,
                data_loader=train_loader,
                epochs=epochs,
                target_epsilon=epsilon,
                target_delta=delta,
                max_grad_norm=MAX_GRAD_NORM,
                # poisson_sampling=False,
            )
        else:
            # print(f"Create private model with noise multiplier {noise_multiplier}")
            private_model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=original_optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=MAX_GRAD_NORM,
                # poisson_sampling=False,
            )
            # print(f"Created private model with noise {optimizer.noise_multiplier}")

        return private_model, optimizer, train_loader, privacy_engine

    @staticmethod
    def get_evaluate_fn(
        test_set,
        dataset_name: str,
        train_parameters: TrainParameters,
        wandb_run: wandb.sdk.wandb_run.Run,
        batch_size: int,
        train_set,
    ) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
        """Return an evaluation function for centralized evaluation."""

        def evaluate(
            server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
        ) -> Optional[Tuple[float, float]]:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = ModelUtils.get_model(dataset_name, device)
            Utils.set_params(model, parameters)
            model.to(device)

            testloader = torch.utils.data.DataLoader(
                test_set,
                batch_size=batch_size,
            )

            (
                test_loss,
                accuracy,
                f1score,
                precision,
                recall,
                max_disparity_test,
            ) = Learning.test(
                model=model,
                test_loader=testloader,
                train_parameters=train_parameters,
                current_epoch=server_round,
            )
            if wandb_run:
                wandb_run.log(
                    {
                        "Centralised Test Loss": test_loss,
                        "Centralised Test Accuracy": accuracy,
                        "Centralised Test F1 Score": f1score,
                        "Centralised Test Precision": precision,
                        "Centralised Test Recall": recall,
                        "Centralised Test Max Disparity": max_disparity_test,
                        "FL Round": server_round,
                    }
                )

            return test_loss, {"Test Accuracy": accuracy}

        return evaluate


class TorchVision_FL(VisionDataset):
    """This is just a trimmed down version of torchvision.datasets.MNIST.

    Use this class by either passing a path to a torch file (.pt)
    containing (data, targets) or pass the data, targets directly
    instead.
    """

    def __init__(
        self,
        path_to_data=None,
        data=None,
        targets=None,
        transform: Optional[Callable] = None,
    ) -> None:
        path = path_to_data.parent if path_to_data else None
        self.dataset_path = path.parent.parent.parent if path_to_data else None

        super(TorchVision_FL, self).__init__(path, transform=transform)
        self.transform = transform

        if path_to_data:
            # load data and targets (path_to_data points to an specific .pt file)
            self.data, self.sensitive_features, self.targets = torch.load(path_to_data)
        else:
            self.data = data
            self.targets = targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if isinstance(img, str):
            path = self.dataset_path / "img_align_celeba/" / self.data[index]
            img = Image.open(path).convert(
                "RGB",
            )

        if not isinstance(img, Image.Image):  # if not PIL image
            if not isinstance(img, np.ndarray):  # if torch tensor
                img = img.numpy()

            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        sensitive_feature = self.sensitive_features[index]

        return img, sensitive_feature, target

    def __len__(self) -> int:
        return len(self.data)
