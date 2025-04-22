import chromadb
from tqdm import tqdm

from engine.data.ms_marco import load_ms_marco

train_ds = load_ms_marco()['train']

chroma_client = chromadb.PersistentClient()

# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="train_docs")

# switch `add` to `upsert` to avoid adding the same documents every time
for i in tqdm(range(len(train_ds))):
    docs = train_ds[i]['passages']['passage_text']
    print(len(docs))
    collection.upsert(
        documents=docs,
        ids=[f'id{i}_{j}' for j in range(len(docs))],
    )
    if i > 300:
        break

results = collection.query(
    query_texts=["This is a query document about florida"], # Chroma will embed this for you
    n_results=2 # how many results to return
)

print(results)

## Make database using all entries in the train dataset
