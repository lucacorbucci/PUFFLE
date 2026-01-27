"""
Generate datasets with distribution shifts for lambda strategy comparison experiments.

This script creates CSV files with controlled distribution shifts that can be
used to test how different lambda update strategies adapt to changing data distributions.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_dutch_dataset(csv_path: str) -> pd.DataFrame:
    """Load the Dutch census dataset."""
    data_path = Path(csv_path) / "dutch_census_2001.csv"
    return pd.read_csv(data_path)


def create_distribution_shift(
    df: pd.DataFrame,
    shift_type: str = "increase_bias",
    shift_magnitude: float = 0.3,
    sensitive_column: str = "sex_binary",
    target_column: str = "occupation_binary",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Create a distribution-shifted version of the dataset.

    Args:
        df: Original dataframe
        shift_type: Type of shift to apply
        shift_magnitude: Proportion of data to modify (0.0 to 1.0)
        sensitive_column: Name of sensitive attribute column
        target_column: Name of target/label column
        seed: Random seed for reproducibility

    Returns:
        Modified dataframe with distribution shift

    """
    np.random.seed(seed)
    df_shifted = df.copy()

    n_samples = len(df_shifted)
    n_to_modify = int(n_samples * shift_magnitude)
    indices_to_modify = np.random.choice(n_samples, n_to_modify, replace=False)

    if shift_type == "increase_bias":
        _augment_increase_bias(
            df_shifted, indices_to_modify, sensitive_column, target_column
        )
    elif shift_type == "decrease_bias":
        _augment_decrease_bias(df_shifted, indices_to_modify, sensitive_column)
    elif shift_type == "flip_labels":
        _augment_flip_labels(df_shifted, indices_to_modify, target_column)
    elif shift_type == "class_imbalance":
        df_shifted = _augment_class_imbalance(df_shifted, n_to_modify, target_column)
    else:
        msg = f"Unknown shift type: {shift_type}"
        raise ValueError(msg)

    return df_shifted


def _augment_increase_bias(
    df_shifted: pd.DataFrame,
    indices_to_modify: np.ndarray,
    sensitive_column: str,
    target_column: str,
) -> None:
    for idx in indices_to_modify:
        target_val = df_shifted.loc[idx, target_column]
        sensitive_val = df_shifted.loc[idx, sensitive_column]

        if target_val == 1 and sensitive_val == 0:
            df_shifted.loc[idx, sensitive_column] = 1
        elif target_val == 0 and sensitive_val == 1:
            df_shifted.loc[idx, sensitive_column] = 0


def _augment_decrease_bias(
    df_shifted: pd.DataFrame,
    indices_to_modify: np.ndarray,
    sensitive_column: str,
) -> None:
    for idx in indices_to_modify:
        current_val = df_shifted.loc[idx, sensitive_column]
        df_shifted.loc[idx, sensitive_column] = 1 - current_val


def _augment_flip_labels(
    df_shifted: pd.DataFrame,
    indices_to_modify: np.ndarray,
    target_column: str,
) -> None:
    for idx in indices_to_modify:
        current_val = df_shifted.loc[idx, target_column]
        df_shifted.loc[idx, target_column] = 1 - current_val


def _augment_class_imbalance(
    df_shifted: pd.DataFrame,
    n_to_modify: int,
    target_column: str,
) -> pd.DataFrame:
    target_to_remove = 0  # Remove class 0 samples
    class_0_indices = df_shifted[df_shifted[target_column] == target_to_remove].index
    indices_to_remove = np.random.choice(
        class_0_indices, min(n_to_modify, len(class_0_indices)), replace=False
    )
    return df_shifted.drop(indices_to_remove).reset_index(drop=True)


def compute_bias_metrics(
    df: pd.DataFrame,
    sensitive_column: str = "sex_binary",
    target_column: str = "occupation_binary",
) -> dict:
    """Compute bias metrics for the dataset."""
    # Demographic parity: P(Y=1|S=1) - P(Y=1|S=0)
    p_y1_s1 = df[df[sensitive_column] == 1][target_column].mean()
    p_y1_s0 = df[df[sensitive_column] == 0][target_column].mean()
    demographic_parity = abs(p_y1_s1 - p_y1_s0)

    # Class balance
    class_balance = df[target_column].mean()

    # Sensitive attribute distribution
    sensitive_balance = df[sensitive_column].mean()

    return {
        "demographic_parity": demographic_parity,
        "class_balance": class_balance,
        "sensitive_balance": sensitive_balance,
        "n_samples": len(df),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate datasets with distribution shifts"
    )

    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to directory containing original Dutch dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to directory where shifted datasets will be saved",
    )
    parser.add_argument(
        "--shift_type",
        type=str,
        default="increase_bias",
        choices=["increase_bias", "decrease_bias", "flip_labels", "class_imbalance"],
        help="Type of distribution shift to apply",
    )
    parser.add_argument(
        "--shift_magnitude",
        type=float,
        default=0.3,
        help="Magnitude of shift (0.0 to 1.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original dataset
    print(f"Loading dataset from {args.input_path}")
    df_original = load_dutch_dataset(args.input_path)

    # Compute original metrics
    print("\nOriginal dataset metrics:")
    original_metrics = compute_bias_metrics(df_original)
    for key, value in original_metrics.items():
        print(
            f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}"
        )

    # Save original dataset
    original_output = output_dir / "dutch_census_2001_original.csv"
    df_original.to_csv(original_output, index=False)
    print(f"\nSaved original dataset to: {original_output}")

    # Create shifted dataset
    print(f"\nApplying {args.shift_type} shift (magnitude={args.shift_magnitude})...")
    df_shifted = create_distribution_shift(
        df_original,
        shift_type=args.shift_type,
        shift_magnitude=args.shift_magnitude,
        seed=args.seed,
    )

    # Compute shifted metrics
    print("\nShifted dataset metrics:")
    shifted_metrics = compute_bias_metrics(df_shifted)
    for key, value in shifted_metrics.items():
        print(
            f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}"
        )

    # Show change
    print("\nChange in metrics:")
    for key in ["demographic_parity", "class_balance", "sensitive_balance"]:
        change = shifted_metrics[key] - original_metrics[key]
        print(f"  Δ{key}: {change:+.4f}")

    # Save shifted dataset
    shifted_output = (
        output_dir
        / f"dutch_census_2001_shifted_{args.shift_type}_{args.shift_magnitude}.csv"
    )
    df_shifted.to_csv(shifted_output, index=False)
    print(f"\nSaved shifted dataset to: {shifted_output}")

    # Save metadata
    metadata = {
        "original_file": str(original_output),
        "shifted_file": str(shifted_output),
        "shift_type": args.shift_type,
        "shift_magnitude": args.shift_magnitude,
        "seed": args.seed,
        "original_metrics": original_metrics,
        "shifted_metrics": shifted_metrics,
    }

    import json

    metadata_output = (
        output_dir / f"shift_metadata_{args.shift_type}_{args.shift_magnitude}.json"
    )
    with metadata_output.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to: {metadata_output}")

    print("\n✅ Dataset generation complete!")
    print("\nTo use these datasets in your experiment:")
    print(f"  --csv_path_before {original_output.parent}")
    print(f"  --csv_path_after {shifted_output.parent}")
    print(f"  --dataset_before {original_output.name}")
    print(f"  --dataset_after {shifted_output.name}")
