"""
Example: Using samples_per_client parameter with FairnessPartitioner

This example demonstrates how to use the new samples_per_client parameter
to control the number of samples each client receives in the FairnessPartitioner.
"""
# ruff: noqa: T201, PERF401, INP001

import pandas as pd
from datasets import Dataset

from FlowerFLTemplate.Datasets.Partitioner.fairness_partitioner import (
    FairnessPartitioner,
)


def main():
    """Demonstrate samples_per_client usage."""
    # Create a sample dataset
    data = []
    for target in [0, 1]:
        for sensitive in [0, 1]:
            for _ in range(250):  # 1000 total samples
                data.append(
                    {
                        "feature1": 0.5,
                        "feature2": 0.5,
                        "target": target,
                        "sensitive": sensitive,
                    }
                )

    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)

    print(f"Total dataset size: {len(dataset)} samples\n")

    # Example 1: Automatic allocation (default behavior)
    print("=" * 60)
    print("Example 1: Automatic allocation (samples_per_client=None)")
    print("=" * 60)

    partitioner_auto = FairnessPartitioner(
        num_partitions=10,
        sensitive_attribute="sensitive",
        target_attribute="target",
        ratio_unfair_clients=0.5,
        group_to_reduce=(1, 1),
        ratio_unfairness=(0.8, 0.9),
        group_to_increment=(1, 0),
        seed=42,
        dataset=dataset,
        samples_per_client=None,  # Automatic allocation
    )

    print("\nClient sample counts (automatic):")
    for i in range(10):
        partition = partitioner_auto.load_partition(i)
        client_type = partitioner_auto.client_types[i]
        print(f"  Client {i} ({client_type:6s}): {len(partition):3d} samples")

    # Example 2: Fixed samples per client
    print("\n" + "=" * 60)
    print("Example 2: Fixed allocation (samples_per_client=80)")
    print("=" * 60)

    partitioner_fixed = FairnessPartitioner(
        num_partitions=10,
        sensitive_attribute="sensitive",
        target_attribute="target",
        ratio_unfair_clients=0.5,
        group_to_reduce=(1, 1),
        ratio_unfairness=(0.8, 0.9),
        group_to_increment=(1, 0),
        seed=42,
        dataset=dataset,
        samples_per_client=80,  # Each client gets ~80 samples
    )

    print("\nClient sample counts (fixed at 80):")
    for i in range(10):
        partition = partitioner_fixed.load_partition(i)
        client_type = partitioner_fixed.client_types[i]
        print(f"  Client {i} ({client_type:6s}): {len(partition):3d} samples")

    # Example 3: Command line usage
    print("\n" + "=" * 60)
    print("Example 3: Command line usage")
    print("=" * 60)
    print("""
To use samples_per_client from the command line:

python src/FlowerFLTemplate/main.py \\
    --num_clients 10 \\
    --num_rounds 5 \\
    --FL_setting cross_device \\
    --dataset_name dutch \\
    --partitioner_type fairness \\
    --sensitive_attribute sex_binary \\
    --target_attribute occupation_binary \\
    --ratio_unfair_clients 0.5 \\
    --group_to_reduce 1 1 \\
    --group_to_increment 1 0 \\
    --ratio_unfairness 0.8 0.9 \\
    --samples_per_client 500 \\
    --fed_dir ./fed_results \\
    ... (other arguments)

This will ensure each client gets approximately 500 samples,
distributed proportionally across groups.
    """)


if __name__ == "__main__":
    main()
