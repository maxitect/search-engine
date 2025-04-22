import pandas as pd
from src.utils.get_df import get_df

# Transform the dataframes
print("Downloading MS MARCO test dataset...")
test_df = get_df(
    "https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/"
    "test-00000-of-00001.parquet"
)
print("Downloading MS MARCO train dataset...")
train_df = get_df(
    "https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/"
    "train-00000-of-00001.parquet"
)
print("Downloading MS MARCO validation dataset...")
val_df = get_df(
    "https://huggingface.co/datasets/microsoft/ms_marco/resolve/main/v1.1/"
    "validation-00000-of-00001.parquet"
)

# Save the transformed dataframes to parquet files
test_df.to_parquet('ms_marco_test.parquet', index=False)
train_df.to_parquet('ms_marco_train.parquet', index=False)
val_df.to_parquet('ms_marco_validation.parquet', index=False)

# Load data from parquet files
df = pd.read_parquet('ms_marco_test.parquet')

print(f"New DataFrame columns: {df.columns.tolist()}")
