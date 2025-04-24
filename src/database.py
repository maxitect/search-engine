import pandas as pd
import pickle
import torch
import tqdm
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
    # Determine the device the model is on
    device = next(model.parameters()).device
    embs = []

    # Create progress bar
    pbar = tqdm(range(0, len(docs), batch_size), desc="Embedding documents")

    for i in pbar:
        batch = docs[i:i+batch_size]
        seqs = []
        for text in batch:
            tokens = preprocess(text)
            ids = [vocab.get(tok, 0) for tok in tokens[:config.MAX_DOC_LEN]]
            # pad to max length
            if len(ids) < config.MAX_DOC_LEN:
                ids += [0] * (config.MAX_DOC_LEN - len(ids))
            seqs.append(ids)
        # Move input tensor to the same device as the model
        x = torch.tensor(seqs, device=device)
        with torch.no_grad():
            emb = model.doc_tower(x)
            # Move embeddings back to CPU for numpy conversion
            emb = emb.cpu()
        # convert to list
        embs.extend(emb.numpy().tolist())
    return embs


def index_to_chroma(docs, embs, collection_name='ms_marco_docs'):
    """Create/get a Chroma collection & add documents with embs in batches."""
    client = Client()
    coll = client.get_or_create_collection(name=collection_name)
    # ChromaDB's max batch size is 5461 (from error message)
    batch_size = 5000  # Setting slightly below limit to be safe

    # Create progress bar
    pbar = tqdm(range(0, len(docs), batch_size), desc="Adding to ChromaDB")

    for i in pbar:
        batch_end = min(i + batch_size, len(docs))
        batch_docs = docs[i:batch_end]
        batch_embs = embs[i:batch_end]
        batch_ids = [str(j) for j in range(i, batch_end)]

        pbar.set_postfix(
            {"batch": f"{i//batch_size + 1}", "docs": len(batch_docs)})
        coll.add(ids=batch_ids, embeddings=batch_embs, documents=batch_docs)

    return coll


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
