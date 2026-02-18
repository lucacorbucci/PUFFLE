"""
Generates the accuracy vs. SP-unfairness tradeoff plot from .npy result files.

Each .npy file has shape (num_clients, 2):
  - column 0: accuracy per client
  - column 1: P1 (Statistical Parity unfairness) per client

The plot sweeps over lambda values to trace the tradeoff curve.
Multiple seeds are averaged to produce mean ± std bands.

Usage:
    uv run python -m competitors.mmd_fair.plot_results \
        --fairfl  /home/lcorbucci/Fair-FL/results/ours \
        --puffle  /path/to/puffle/results \
        --dataset compas \
        --output  tradeoff_compas.png
"""

import argparse
import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def load_results(
    results_dir: str, dataset: str
) -> dict[float, list[tuple[float, float]]]:
    """
    Load all .npy result files for a dataset from a directory.

    Groups files by lambda value and returns a dict mapping each lambda to
    a list of (mean_acc, mean_p1) tuples — one per seed.

    Args:
        results_dir: Directory containing .npy files
        dataset: Dataset name prefix (e.g. 'compas')

    Returns:
        Dict mapping lambda -> list of (accuracy, p1) per seed
    """
    pattern = re.compile(rf"^{re.escape(dataset)}_p_([0-9eE+\-\.]+)_(\d+)_(\d+)\.npy$")

    # Group files: lambda -> list of (seed, filepath)
    lambda_to_files: dict[float, list[tuple[int, str]]] = defaultdict(list)

    for fname in os.listdir(results_dir):
        m = pattern.match(fname)
        if m:
            lambda_val = float(m.group(1))
            seed = int(m.group(2))
            lambda_to_files[lambda_val].append((seed, os.path.join(results_dir, fname)))

    # For each lambda, compute (acc, p1) averaged across clients, per seed
    results: dict[float, list[tuple[float, float]]] = {}
    for lambda_val, seed_files in sorted(lambda_to_files.items()):
        seed_points = []
        for seed, fpath in sorted(seed_files):
            data = np.load(fpath)  # shape: (num_clients, 2)
            # Simple mean across clients (Fair-FL uses equal weights here)
            acc = float(data[:, 0].mean())
            p1 = float(data[:, 1].mean())
            seed_points.append((acc, p1))
        results[lambda_val] = seed_points

    return results


def compute_curve(
    results: dict[float, list[tuple[float, float]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean ± std tradeoff curve from per-seed results.

    Args:
        results: Dict mapping lambda -> list of (acc, p1) per seed

    Returns:
        (p1_mean, p1_std, acc_mean, acc_std) arrays sorted by p1_mean
    """
    lambdas = sorted(results.keys())
    p1_means, p1_stds, acc_means, acc_stds = [], [], [], []

    for lam in lambdas:
        points = results[lam]
        accs = np.array([p[0] for p in points])
        p1s = np.array([p[1] for p in points])
        p1_means.append(p1s.mean())
        p1_stds.append(p1s.std())
        acc_means.append(accs.mean())
        acc_stds.append(accs.std())

    # Sort by P1 (x-axis) for a clean curve
    order = np.argsort(p1_means)
    return (
        np.array(p1_means)[order],
        np.array(p1_stds)[order],
        np.array(acc_means)[order],
        np.array(acc_stds)[order],
    )


def plot_tradeoff(
    curves: list[tuple[str, str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    output_path: str,
    title: str = "COMPAS dataset",
) -> None:
    """
    Plot accuracy vs. SP-unfairness tradeoff curves.

    Args:
        curves: List of (label, color, p1_mean, p1_std, acc_mean, acc_std)
        output_path: Path to save the plot
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for label, color, p1_mean, p1_std, acc_mean, acc_std in curves:
        ax.plot(p1_mean, acc_mean, label=label, color=color, linewidth=2)
        ax.fill_between(
            p1_mean,
            acc_mean - acc_std,
            acc_mean + acc_std,
            alpha=0.15,
            color=color,
        )

    ax.set_xlabel("SP Unfairness", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(f"(b) {title}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    plt.close()


def main() -> None:
    """Entry point for the plotting script."""
    parser = argparse.ArgumentParser(
        description="Plot accuracy vs. SP-unfairness tradeoff"
    )
    parser.add_argument(
        "--fairfl",
        default=None,
        help="Path to Fair-FL results directory (e.g. /home/.../Fair-FL/results/ours)",
    )
    parser.add_argument(
        "--puffle",
        default=None,
        help="Path to PUFFLE MMD-Fair results directory",
    )
    parser.add_argument(
        "--dataset",
        default="compas",
        help="Dataset name prefix used in filenames (default: compas)",
    )
    parser.add_argument(
        "--output",
        default="tradeoff_plot.png",
        help="Output file path for the plot (default: tradeoff_plot.png)",
    )
    parser.add_argument(
        "--title",
        default="COMPAS dataset",
        help="Plot title (default: 'COMPAS dataset')",
    )

    args = parser.parse_args()

    if args.fairfl is None and args.puffle is None:
        parser.error("At least one of --fairfl or --puffle must be specified")

    curves = []

    if args.fairfl:
        print(f"Loading Fair-FL results from: {args.fairfl}")
        fairfl_results = load_results(args.fairfl, args.dataset)
        print(f"  Found {len(fairfl_results)} lambda values")
        p1_mean, p1_std, acc_mean, acc_std = compute_curve(fairfl_results)
        curves.append(("Fair-FL (Ours)", "#1f4e79", p1_mean, p1_std, acc_mean, acc_std))

    if args.puffle:
        print(f"Loading PUFFLE results from: {args.puffle}")
        puffle_results = load_results(args.puffle, args.dataset)
        print(f"  Found {len(puffle_results)} lambda values")
        p1_mean, p1_std, acc_mean, acc_std = compute_curve(puffle_results)
        curves.append(
            ("PUFFLE MMD-Fair", "#c0392b", p1_mean, p1_std, acc_mean, acc_std)
        )

    plot_tradeoff(curves, args.output, title=args.title)

    # Print summary stats
    for label, _, p1_mean, _, acc_mean, _ in curves:
        print(f"\n{label}:")
        print(f"  P1 range:  [{p1_mean.min():.4f}, {p1_mean.max():.4f}]")
        print(f"  Acc range: [{acc_mean.min():.4f}, {acc_mean.max():.4f}]")


if __name__ == "__main__":
    main()
