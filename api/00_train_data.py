import argparse
import pandas as pd
import requests
from src.utils.get_df import get_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='For deployment only save docs')
    parser.add_argument(
        '--deploy',
        action='store_true',
        help='Only save ms_marco_docs for deployment'
    )
    args = parser.parse_args()
    # Download and save text8 corpus
    print("Downloading text8 corpus...")
    r = requests.get(
        'https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8')
    with open('text8', 'wb') as f:
        f.write(r.content)

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

    df = pd.concat([train_df, test_df, val_df], ignore_index=True)
    print(f"New DataFrame columns: {df.columns.tolist()}")

    print("Retrieving all individual MS MARCO documents...")
    ms_marco_docs = []
    passage_count = 0
    for passages in df.documents:
        for passage in passages:
            passage_count += 1
            ms_marco_docs.append(passage)
    docs_df = pd.DataFrame(ms_marco_docs)

    print("Saving documents...")
    if not args.deploy:
        test_df.to_parquet('ms_marco_test.parquet', index=False)
        train_df.to_parquet('ms_marco_train.parquet', index=False)
        val_df.to_parquet('ms_marco_validation.parquet', index=False)
    docs_df.to_parquet('ms_marco_docs.parquet', index=False)
