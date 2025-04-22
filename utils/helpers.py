import faiss
import numpy as np
import torch

def build_faiss_index(document_embeddings):
    """Build FAISS index for efficient similarity search"""
    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
    index.add(document_embeddings)
    return index

def search_documents(query_embedding, index, document_ids, k=5):
    """Search for top-k similar documents"""
    distances, indices = index.search(query_embedding, k)
    return [(document_ids[i], distances[0][j]) for j, i in enumerate(indices[0])]

def save_embeddings(embeddings, ids, path):
    """Save embeddings and their IDs"""
    np.savez(path, embeddings=embeddings, ids=ids)

def load_embeddings(path):
    """Load saved embeddings"""
    data = np.load(path)
    return data['embeddings'], data['ids']