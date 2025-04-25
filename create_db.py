import argparse
import logging  # Add logging

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from tqdm import tqdm

from engine.data.ms_marco import load_ms_marco
from inference import setup_semantics_embedder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig() # Basic config for logging

CONFIG_OPTIONS = ['chroma','my_embeddings', 'gensim_embeddings', 'baseline']

# --- Helper function to parse arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description='Create ChromaDB database for MSMarco')
    parser.add_argument(
        '--config', type=str, required=True, choices=CONFIG_OPTIONS,
        help='Embedding configuration to use for creating the database.'
    )
    parser.add_argument(
        '--device', type=str, default='cpu', choices=['cpu', 'gpu', 'cuda'],
        help='Device to run inference on (cpu or gpu/cuda).'
    )
    parser.add_argument(
        '--num_entries', type=int, default=None,
        help='Number of entries from MSMarco dataset to process (default: all).'
    )
    args = parser.parse_args()
    # Normalize 'gpu' to 'cuda'
    if args.device == 'gpu':
        args.device = 'cuda'
    return args


def create_db_chroma_embeddings(num_entries=None, device='cuda'):
    # Set up Chroma
    chroma_client = chromadb.PersistentClient()
    


def create_db_my_embeddings(config, device, num_entries=None):
    chroma_client = chromadb.PersistentClient()
    train_ds = load_ms_marco()['train']

    if num_entries is None:
        num_entries = len(train_ds)

    if config == 'chroma':
        # Get Chroma embedding function
        ef = SentenceTransformerEmbeddingFunction(
            model_name='thenlper/gte-small', device=device,
        )

        collection = chroma_client.get_or_create_collection(
            name=f'train_docs_{config}',
            embedding_function=ef,
        )

        # Add data
        # Batch add documents to the collection
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
    else: 
        semantics_embedder = setup_semantics_embedder(config)
        
        collection = chroma_client.get_or_create_collection(
            name=f'train_docs_{config}',
        )

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
    args = parse_args() # Parse arguments

    if args.config == 'chroma':
        create_db_chroma_embeddings(num_entries=args.num_entries)
    else:
        # Pass device and num_entries for custom embeddings
        create_db_my_embeddings(
            config=args.config,
            device=args.device,
            num_entries=args.num_entries
        )

    logger.info("Database creation process finished.")


