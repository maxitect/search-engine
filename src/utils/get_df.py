from io import BytesIO
import pandas as pd
import requests


def get_df(url):
    r = requests.get(url)
    df = pd.read_parquet(BytesIO(r.content))
    print(f"Previous DataFrame columns: {df.columns.tolist()}")

    documents = []
    labels = []
    queries = []

    for (passages, query) in zip(df['passages'], df['query']):
        documents.append(passages['passage_text'])
        labels.append(passages['is_selected'])
        queries.append(query)

    return pd.DataFrame({
        'documents': documents,
        'labels': labels,
        'queries': queries
    })
