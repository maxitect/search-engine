from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb
from tqdm import tqdm

from engine.data.ms_marco import load_ms_marco
from inference import setup_semantics_embedder


def create_db_chroma_embeddings():
    ef = SentenceTransformerEmbeddingFunction(
        model_name='thenlper/gte-small', device='cuda',
    )

    # ef = embedding_functions.ONNXMiniLM_L6_V2(
    #     # model_name="all-MiniLM-L6-v2",
    #     preferred_providers=['CUDAExecutionProvider']
    # )

    train_ds = load_ms_marco()['train']

    chroma_client = chromadb.PersistentClient()

    # switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
    collection = chroma_client.get_or_create_collection(
        name='train_docs',
        embedding_function=ef,
    )

    # Batch add documents to the collection
    batch_size = 512
    for i in tqdm(range(0, len(train_ds), batch_size)):
        docs = []
        ids = []
        for j in range(batch_size):
            if i+j >= len(train_ds):
                break
            docs_row = train_ds[i+j]['passages']['passage_text']
            docs += docs_row
            ids += [f'id{i+j}_{k}' for k in range(len(docs_row))]
        collection.upsert(
            documents=docs,
            ids=ids,
        )

    results = collection.query(
        # Chroma will embed this for you
        query_texts=['This is a query document about florida'],
        n_results=2,  # how many results to return
    )

    print(results)


def create_db_my_embeddings():
    semantics_embedder = setup_semantics_embedder()
    chroma_client = chromadb.PersistentClient()
    collection = chroma_client.get_or_create_collection(
        name='train_docs_my_embeddings',
    )

    train_ds = load_ms_marco()['train']

    for i in tqdm(range(2000)):
        docs_row = train_ds[i]['passages']['passage_text']
        embeddings = [
            semantics_embedder.embed_doc(
                d,
            ).tolist() for d in docs_row
        ]
        ids = [f'id{i}_{k}' for k in range(len(docs_row))]
        collection.upsert(
            documents=docs_row,
            embeddings=embeddings,
            ids=ids,
        )


if __name__ == '__main__':
    # create_db_chroma_embeddings()
    create_db_my_embeddings()
