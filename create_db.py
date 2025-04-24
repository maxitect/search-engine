from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import chromadb
from tqdm import tqdm

from engine.data.ms_marco import load_ms_marco
from inference import setup_semantics_embedder

CONFIG_OPTIONS = ['my_embeddings', 'gensim_embeddings', 'baseline']


def create_db_chroma_embeddings(num_entries=None):
    # Set up Chroma
    chroma_client = chromadb.PersistentClient()
    # Get Chroma embedding function
    ef = SentenceTransformerEmbeddingFunction(
        model_name='thenlper/gte-small', device='cuda',
    )

    collection = chroma_client.get_or_create_collection(
        name='train_docs',
        embedding_function=ef,
    )

    # Add data
    train_ds = load_ms_marco()['train']
    # Batch add documents to the collection

    if num_entries is None:
        num_entries = len(train_ds)

    # Maually selected size before it gives chromadb.errors.InternalError
    batch_size = 512
    for i in tqdm(range(0, num_entries, batch_size)):
        docs = []
        ids = []
        for j in range(batch_size):
            if i+j >= num_entries:
                break
            docs_row = train_ds[i+j]['passages']['passage_text']
            docs += docs_row
            ids += [f'id{i+j}_{k}' for k in range(len(docs_row))]
        collection.upsert(
            documents=docs,
            ids=ids,
        )


def create_db_my_embeddings(config, num_entries=None):

    semantics_embedder = setup_semantics_embedder(config)
    chroma_client = chromadb.PersistentClient()
    collection = chroma_client.get_or_create_collection(
        name=f'train_docs_{config}2',
    )

    train_ds = load_ms_marco()['train']

    if num_entries is None:
        num_entries = len(train_ds)

    for i in tqdm(range(num_entries)):
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
    # config = 'my_embeddings'
    # config = 'gensim_embeddings'
    # create_db_my_embeddings(2000)
    create_db_my_embeddings('baseline', 2000)
