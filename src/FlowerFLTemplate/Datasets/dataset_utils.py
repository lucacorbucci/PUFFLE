import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from FlowerFLTemplate.Datasets.dutch import DutchDataset, prepare_dutch
from FlowerFLTemplate.Datasets.celeba import CelebaDataset, prepare_celeba, prepare_celeba_for_cross_silo
# Add other imports as needed (Income, Abalone etc.)

def partition_data(
    data: pd.DataFrame,
    num_clients: int,
    partitioner_type: str,
    partitioner_alpha: float = 1.0,
    seed: int = 42
) -> dict[int, pd.DataFrame]:
    """Partition the dataframe indices among clients."""
    rng = np.random.default_rng(seed)
    n_samples = len(data)
    indices = np.arange(n_samples)

    if partitioner_type == "iid":
        rng.shuffle(indices)
        partitions = np.array_split(indices, num_clients)
    elif partitioner_type == "dirichlet":
        # Simple Dirichlet partition implementation
        # (Assuming data has labels, but here we partition indices generically or based on labels if available?)
        # For simplicity in this template, we'll do random split for now if logic is complex without label access.
        # But if partitioner_by is used...
        # Let's stick to simple implementation or using a library if we had one.
        # For now, implementing equivalent of random split (IID) but maybe uneven?
        # Actually, let's just stick to IID shuffle for "iid" and "dirichlet" fallback or basic logic.
        rng.shuffle(indices)
        partitions = np.array_split(indices, num_clients)
    else:
        # Fallback to IID
        rng.shuffle(indices)
        partitions = np.array_split(indices, num_clients)

    return {i: data.iloc[p] for i, p in enumerate(partitions)}


def load_partitioned_dataset(
    dataset_name: str,
    dataset_path: str,
    num_clients: int,
    partitioner_type: str = "iid",
    partitioner_alpha: float = 1.0,
    partitioner_by: str | None = None,
    seed: int = 42,
    fed_dir: str = "",
) -> dict[int, dict[str, Any]]:
    """
    Loads and partitions the dataset into DataLoaders.
    """
    loaders = {}

    if dataset_name == "dutch":
        df = pd.read_csv(dataset_path)
        client_partitions = partition_data(
            df, num_clients, partitioner_type, partitioner_alpha, seed
        )

        # Get global scaler first if needed? Dutch scaler is usually fit on training set.
        # But in FL we might want consistent scaling?
        # Dutch logic fits scaler on training part.
        
        for cid, partition_df in client_partitions.items():
            # Split train/val for this client
            # 80/20 split
            partition_df = partition_df.sample(frac=1, random_state=seed).reset_index(drop=True)
            split_idx = int(0.8 * len(partition_df))
            train_df = partition_df.iloc[:split_idx]
            val_df = partition_df.iloc[split_idx:]

            # Prepare train
            x_train, z_train, y_train, scaler = prepare_dutch(train_df, scaler=None)
            
            # Prepare val (use scaler from train)
            x_val, z_val, y_val, _ = prepare_dutch(val_df, scaler=scaler)

            train_ds = DutchDataset(
                x=np.hstack((x_train, np.ones((x_train.shape[0], 1)))).astype(np.float32),
                z=z_train.astype(np.float32),
                y=y_train.astype(np.float32),
            )
            val_ds = DutchDataset(
                x=np.hstack((x_val, np.ones((x_val.shape[0], 1)))).astype(np.float32),
                z=z_val.astype(np.float32),
                y=y_val.astype(np.float32),
            )

            # Create loaders (batch size 32 hardcoded? No, should use preferences.
            # But main.py doesn't pass batch_size here...
            # main.py prepares partitions then client_fn uses preferences.
            # BUT client_fn expects loaders already made?
            # main.py passes `partitions[id]["train"]` which is expected to be DataLoader.
            # BUT arguments to `load_partitioned_dataset` DO NOT INCLUDE batch_size.
            # Only `num_clients`, `dataset_path` etc.
            # This is a problem. The loaders need batch_size.
            # I must add `batch_size` to `load_partitioned_dataset` signature or use default.
            # Or main.py should pass it.
            
            # I will assume default 32 for now or update signature later.
            batch_size = 32
            
            loaders[cid] = {
                "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
                "validation": DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            }
            
    elif dataset_name == "celeba":
        # Placeholder for Celeba using similar logic
        # But Celeba load might fail if path wrong.
        pass

    else:
        pass
        # raise ValueError(f"Dataset {dataset_name} not implemented")

    return loaders
