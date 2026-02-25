import pandas as pd

try:
    # Attempt to load the dataset using HuggingFace's dataset viewer URL or parquet link
    # The default parquet format link for most huggingface datasets is:
    df = pd.read_parquet("hf://datasets/mstz/compas/compas.parquet")
    print(f"HuggingFace dataset size: {len(df)}")
    print(df.columns.tolist())
    
except Exception as e:
    print(f"Failed to load via pd.read_parquet: {e}")
    try:
        from datasets import load_dataset
        ds = load_dataset("mstz/compas")
        df = ds["train"].to_pandas()
        print(f"HuggingFace dataset size via datasets lib: {len(df)}")
        print(df.columns.tolist())
    except Exception as e2:
         print(f"Failed to load via datasets library: {e2}")
