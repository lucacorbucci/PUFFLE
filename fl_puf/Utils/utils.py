import json
import shutil
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import wandb
from PIL import Image
from flwr.common.typing import Scalar
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset

from DPL.RegularizationLoss import RegularizationLoss
from DPL.Utils.model_utils import ModelUtils
from DPL.learning import Learning
from fl_puf.FederatedDataset.PartitionTypes.balanced_and_unbalanced import (
    BalancedAndUnbalanced,
)
from fl_puf.FederatedDataset.PartitionTypes.iid_partition import IIDPartition
from fl_puf.FederatedDataset.PartitionTypes.non_iid_partition_with_sensitive_feature import (
    NonIIDPartitionWithSensitiveFeature,
)
from fl_puf.FederatedDataset.PartitionTypes.unbalanced_partition import (
    UnbalancedPartition,
)
from fl_puf.FederatedDataset.PartitionTypes.unbalanced_partition_one_class import (
    UnbalancedPartitionOneClass,
)
from fl_puf.FederatedDataset.PartitionTypes.underrepresented_partition import (
    UnderrepresentedPartition,
)
from fl_puf.FederatedDataset.Utils.utils import PartitionUtils
from fl_puf.Utils.train_parameters import TrainParameters


class Utils:
    @staticmethod
    def setup_wandb(args, train_parameters):
        if train_parameters.noise_multiplier > 0:
            noise_multiplier = train_parameters.noise_multiplier
        elif args.noise_multiplier > 0:
            noise_multiplier = args.noise_multiplier
        else:
            noise_multiplier = 0
        wandb_run = wandb.init(
            # set the wandb project where this run will be logged
            project="FL_fairness",
            # name=f"FL - Lambda {args.DPL_lambda} - LR {args.lr} - Batch {args.batch_size}",
            # track hyperparameters and run metadata
            config={
                "learning_rate": args.lr,
                "csv": args.train_csv,
                "DPL_regularization": args.DPL,
                # "DPL_lambda": args.DPL_lambda,
                "batch_size": args.batch_size,
                "dataset": args.dataset,
                "num_rounds": args.num_rounds,
                "pool_size": args.pool_size,
                "sampled_clients": args.sampled_clients,
                "epochs": args.epochs,
                "private": args.private,
                "epsilon": args.epsilon if args.private else None,
                "gradnorm": args.clipping,
                "delta": args.delta if args.private else 0,
                "noise_multiplier": noise_multiplier,
                "probability_estimation": args.probability_estimation,
                "perfect_probability_estimation": args.perfect_probability_estimation,
                "alpha": args.alpha,
                "percentage_unbalanced_nodes": args.percentage_unbalanced_nodes,
                "alpha_target_lambda": args.alpha_target_lambda,
                "target": args.target,
                "weight_decay_lambda": args.weight_decay_lambda,
                "starting_lambda_mode": args.starting_lambda_mode,
                "starting_lambda_value": args.starting_lambda_value,
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
        disparities = []
        for node in nodes:
            max_disparity = np.max(
                [
                    RegularizationLoss().compute_violation_with_argmax(
                        predictions_argmax=[sample["y"] for sample in node]
                        if isinstance(node, list)
                        else node["y"],
                        sensitive_attribute_list=[sample["z"] for sample in node]
                        if isinstance(node, list)
                        else node["z"],
                        current_target=target,
                        current_sensitive_feature=sv,
                    )
                    for target in range(0, 1)
                    for sv in range(0, 1)
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
        elif dataset == "adult":
            return torch.load(path_to_data)
        elif dataset == "german":
            return torch.load(path_to_data)
        elif dataset == "compas":
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
    ):
        """Torchvision (e.g. CIFAR-10) datasets using LDA."""
        print("Partitioning the dataset")

        images, sensitive_attribute, labels = torch.load(path_to_dataset)
        mapping = {-1: 0, 1: 1}
        sensitive_attribute = torch.tensor(
            [mapping[item] for item in sensitive_attribute]
        )

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
        elif partition_type == "unbalanced":
            # This is the partition type where we assign to each node only
            # a subset of the classes. For instance, if we have 2 classes and
            # 2 sensitive attributes (male and female) like in Celeba, we would assign
            # only the female samples to a node and only the male samples to another node.
            ratio = train_parameters.percentage_unbalanced_nodes
            ratio_list = [ratio, 1 - ratio]
            partitions_index_list = UnbalancedPartition.do_partitioning(
                labels=labels,
                sensitive_features=sensitive_attribute,
                num_partitions=pool_size,
                total_num_classes=2,
                alpha=alpha,
                ratio_list=ratio_list,
            )
            partitions = PartitionUtils.create_splitted_dataset_from_tuple(
                splitted_indexes=partitions_index_list,
                dataset=dataset,
            )
        elif partition_type == "unbalanced_one_class":
            # This partition type allows us to assign to some nodes
            # only 3 out of 4 groups. For instances, if we have 2 classes
            # and 2 sensitive attributes, we want that some nodes have
            # all the possible combinations of classes and sensitive attributes
            # except for some other nodes that will only have 3 out of 4.

            # Ratio is the percentage of nodes that will not have all the possible
            # combinations of classes and sensitive attributes.
            ratio = train_parameters.percentage_unbalanced_nodes

            partitions_index_list = UnbalancedPartitionOneClass.do_partitioning(
                labels=labels,
                sensitive_features=sensitive_attribute,
                num_partitions=pool_size,
                total_num_classes=2,
                alpha=alpha,
                ratio=ratio,
            )
            partitions = PartitionUtils.create_splitted_dataset_from_tuple(
                splitted_indexes=partitions_index_list,
                dataset=dataset,
            )
        elif partition_type == "underrepresented":
            ratio = train_parameters.percentage_unbalanced_nodes

            partitions_index_list = UnderrepresentedPartition.do_partitioning(
                labels=labels,
                sensitive_features=sensitive_attribute,
                num_partitions=pool_size,
                total_num_classes=2,
                alpha=alpha,
                ratio=ratio,
            )
            partitions = PartitionUtils.create_splitted_dataset_from_tuple(
                splitted_indexes=partitions_index_list,
                dataset=dataset,
            )
        elif partition_type == "balanced_and_unbalanced":
            partitions_index_list, metadata = BalancedAndUnbalanced.do_partitioning(
                labels=labels,
                sensitive_features=sensitive_attribute,
                num_partitions=pool_size,
                total_num_classes=2,
                alpha=alpha,
                percentage_unbalanced_nodes=train_parameters.percentage_unbalanced_nodes,
                unbalanced_ratio=train_parameters.unbalanced_ratio,
            )
            partitions = PartitionUtils.create_splitted_dataset_from_tuple(
                splitted_indexes=partitions_index_list,
                dataset=dataset,
            )
        else:
            raise ValueError(f"Unknown partition type: {partition_type}")

        for p in range(pool_size):
            partition_zero = partitions[p][2]
            hist, _ = np.histogram(partition_zero, bins=list(range(num_classes + 1)))
            print(
                f"Class histogram for {p}-th partition, {num_classes} classes): {hist}"
            )

            partition_zero = partitions[p][1]

            hist_sv, _ = np.histogram(partition_zero, bins=list(range(num_classes + 1)))
            print(
                f"Sensitive Value histogram for {p}-th partition, {num_classes} classes): {hist_sv}"
            )
            assert sum(hist) == sum(hist_sv)

            labels_and_sensitive = zip(
                [
                    item.item() if isinstance(item, torch.Tensor) else item
                    for item in partitions[p][2]
                ],
                [
                    item.item() if isinstance(item, torch.Tensor) else item
                    for item in partitions[p][1]
                ],
            )
            counter = Counter(labels_and_sensitive)
            print(
                f"Node has {sum(counter.values())} - ",
                counter,
            )

        # now save partitioned dataset to disk
        # first delete dir containing splits (if exists), then create it
        splits_dir = path_to_dataset.parent / "federated"
        if splits_dir.exists() and partition == "train":
            shutil.rmtree(splits_dir)
            Path.mkdir(splits_dir, parents=True)

        nodes = []
        datasets = []
        for p in range(pool_size):
            labels = partitions[p][2]
            sensitive_features = partitions[p][1]

            image_idx = partitions[p][0]
            imgs = [images[image_id] for image_id in image_idx]

            # create dir
            if not (splits_dir / str(p)).exists():
                Path.mkdir(splits_dir / str(p))

            # if val_ratio > 0.0:
            #     # split data according to val_ratio
            #     train_idx, val_idx = Utils.get_random_id_splits(len(labels), val_ratio)
            #     val_imgs = [imgs[val_id] for val_id in val_idx]
            #     val_labels = labels[val_idx]
            #     val_sensitive = sensitive_features[val_idx]

            #     with open(splits_dir / str(p) / "val.pt", "wb") as f:
            #         torch.save([val_imgs, val_sensitive, val_labels], f)

            #     a = torch.load(splits_dir / str(p) / "val.pt")

            #     imgs = [imgs[train_id] for train_id in train_idx]
            #     labels = labels[train_idx]
            #     sensitive_features = sensitive_features[train_idx]
            nodes.append({"y": labels, "z": sensitive_features})
            with open(
                splits_dir
                / str(p)
                / ("train.pt" if partition == "train" else "test.pt"),
                "wb",
            ) as f:
                torch.save([imgs, sensitive_features, labels], f)

        disparities = Utils.compute_disparities_debug(nodes)

        # store statistics about the dataset in the same folder
        statistics = Utils.get_dataset_statistics_with_lists(
            nodes, disparities, metadata
        )
        for statistic, p in zip(statistics, range(pool_size)):
            with open(
                splits_dir / str(p) / ("metadata.json"),
                "w",
            ) as outfile:
                json_object = json.dumps(statistic, indent=4)
                outfile.write(json_object)
        return splits_dir

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
                np.array(dataset.gender),
                np.array(dataset.targets),
            ],
            train_path,
        )

        print("Data Correctly downloaded")

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
            else "test"
            if partition == "test"
            else "val"
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
            )
        else:
            print("Create private model with noise multiplier")
            private_model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=original_optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=MAX_GRAD_NORM,
            )

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
