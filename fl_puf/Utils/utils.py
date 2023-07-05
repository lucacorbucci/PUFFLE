import shutil
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch
import wandb
from DPL.learning import Learning
from DPL.Utils.model_utils import ModelUtils
from DPL.Utils.train_parameters import TrainParameters
from FederatedDataset.PartitionTypes.iid_partition import IIDPartition
from FederatedDataset.Utils.utils import PartitionUtils
from flwr.common.typing import Scalar
from opacus import PrivacyEngine
from opacus.grad_sample import GradSampleModule
from opacus.optimizers import DPOptimizer
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VisionDataset


class Utils:
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
            raise ValueError(f"Unknown dataset: {dataset_name}")

    @staticmethod
    def get_dataset(path_to_data: Path, cid: str, partition: str, dataset: str):
        # generate path to cid's data
        path_to_data = path_to_data / cid / (partition + ".pt")

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
        path_to_dataset,
        pool_size,
        alpha,
        num_classes,
        partition_type: str,
        val_ratio=0.0,
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

        if partition_type == "iid":
            splitted_indexes = IIDPartition.do_iid_partitioning_with_indexes(
                indexes=idx,
                num_partitions=pool_size,
            )
            partitions = PartitionUtils.create_splitted_dataset_from_tuple(
                splitted_indexes=splitted_indexes,
                dataset=dataset,
            )

        for p in range(pool_size):
            partition_zero = partitions[p][2]
            hist, _ = np.histogram(partition_zero, bins=list(range(num_classes + 1)))
            print(
                f"Class histogram for {p}-th partition (alpha={alpha}, {num_classes} classes): {hist}"
            )

            partition_zero = partitions[p][1]

            hist_sv, _ = np.histogram(partition_zero, bins=list(range(num_classes + 1)))
            print(
                f"Sensitive Value histogram for {p}-th partition (alpha={alpha}, {num_classes} classes): {hist_sv}"
            )
            assert sum(hist) == sum(hist_sv)

        # now save partitioned dataset to disk
        # first delete dir containing splits (if exists), then create it
        splits_dir = path_to_dataset.parent / "federated"
        if splits_dir.exists():
            shutil.rmtree(splits_dir)
        Path.mkdir(splits_dir, parents=True)

        for p in range(pool_size):
            labels = partitions[p][2]
            sensitive_features = partitions[p][1]

            image_idx = partitions[p][0]
            imgs = [images[image_id] for image_id in image_idx]

            # create dir
            Path.mkdir(splits_dir / str(p))

            if val_ratio > 0.0:
                # split data according to val_ratio
                train_idx, val_idx = Utils.get_random_id_splits(len(labels), val_ratio)
                val_imgs = [imgs[val_id] for val_id in val_idx]
                val_labels = labels[val_idx]
                val_sensitive = sensitive_features[val_idx]

                with open(splits_dir / str(p) / "val.pt", "wb") as f:
                    torch.save([val_imgs, val_sensitive, val_labels], f)

                a = torch.load(splits_dir / str(p) / "val.pt")

                imgs = [imgs[train_id] for train_id in train_idx]
                labels = labels[train_idx]
                sensitive_features = sensitive_features[train_idx]

            with open(splits_dir / str(p) / "train.pt", "wb") as f:
                torch.save([imgs, sensitive_features, labels], f)

        return splits_dir

    @staticmethod
    def prepare_dataset_for_FL(
        train_set,
        base_path: str,
        dataset_name: str,
    ):
        # fuse all data splits into a single "training.pt"
        data_loc = Path(base_path) / f"{dataset_name}/{dataset_name}-10-batches-py"
        train_path = data_loc / "training.pt"
        print("Generating unified dataset")
        torch.save(
            [
                train_set.samples,
                np.array(train_set.gender),
                np.array(train_set.targets),
            ],
            train_path,
        )

        print("Data Correctly downloaded")

        return train_path

    @staticmethod
    def get_dataloader(
        path_to_data: str,
        cid: str,
        is_train: bool,
        batch_size: int,
        workers: int,
        dataset: str,
    ):
        """Generates trainset/valset object and returns appropiate dataloader."""

        partition = "train" if is_train else "val"
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
            private_model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=original_optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=MAX_GRAD_NORM,
            )

        return private_model, optimizer, train_loader

    @staticmethod
    def get_evaluate_fn(
        test_set,
        dataset_name: str,
        train_parameters: TrainParameters,
        wandb_run: wandb.sdk.wandb_run.Run,
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
                batch_size=128,
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

            # if wandb_run:
            #     wandb_run.log(
            #         {
            #             "epoch": server_round,
            #             "Test Accuracy": accuracy,
            #             "Test Loss": test_loss,
            #         },
            #     )

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
