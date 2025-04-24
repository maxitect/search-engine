import pandas as pd
import pickle
import torch
from chromadb import Client

import src.config as config
from src.utils.tokenise import preprocess
from src.utils.model_loader import load_twotowers


def load_docs(path):
    """Load passages from a parquet file and return a list of strings."""
    df = pd.read_parquet(path)
    # assume single-column dataframe of texts
    return df.values.flatten().tolist()


def embed_docs(model, vocab, docs, batch_size=64):
    """Embed documents in batches using the document tower."""
    embs = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        seqs = []
        for text in batch:
            tokens = preprocess(text)
            ids = [vocab.get(tok, 0) for tok in tokens[:config.MAX_DOC_LEN]]
            # pad to max length
            if len(ids) < config.MAX_DOC_LEN:
                ids += [0] * (config.MAX_DOC_LEN - len(ids))
            seqs.append(ids)
        x = torch.tensor(seqs)
        with torch.no_grad():
            emb = model.doc_tower(x)
        # convert to list
        embs.extend(emb.numpy().tolist())
    return embs


def index_to_chroma(docs, embs, collection_name='ms_marco_docs'):
    """Create or get a ChromaDB collection & add documents with embeddings."""
    client = Client()
    coll = client.get_or_create_collection(name=collection_name)
    ids = [str(i) for i in range(len(docs))]
    coll.add(ids=ids, embeddings=embs, documents=docs)


def build_chroma(
        path='ms_marco_docs.parquet',
        collection_name='ms_marco_docs',
        batch_size=64
):
    """Full pipeline: load docs, load model, embed and index to ChromaDB."""
    vocab_to_int = pickle.load(open(config.VOCAB_TO_ID_PATH, 'rb'))
    docs = load_docs(path)
    model = load_twotowers(vocab_to_int)
    embs = embed_docs(model, vocab_to_int, docs, batch_size)
    index_to_chroma(docs, embs, collection_name)
