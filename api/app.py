from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
import torch
import pickle
from chromadb import Client
from contextlib import asynccontextmanager

import src.config as config
from src.utils.tokenise import preprocess
from src.utils.model_loader import load_twotowers

# Define response models


class SearchResult(BaseModel):
    id: str
    document: str
    similarity: float


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]

# Setup application startup and shutdown


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models and vocabulary on startup
    app.state.vocab_to_int = pickle.load(open(config.VOCAB_TO_ID_PATH, 'rb'))
    app.state.model = load_twotowers(app.state.vocab_to_int)
    app.state.chroma_client = Client()
    app.state.collection = app.state.chroma_client.get_collection(
        name="ms_marco_docs")
    print("Search engine initialized and ready")
    yield
    # Clean up resources on shutdown
    print("Shutting down search engine")

# Create FastAPI app
app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Search engine API is running"}


@app.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, description="Search query"),
    top_k: int = Query(
        5, ge=1, le=20, description="Number of results to return")
):
    # Preprocess and encode query
    device = next(app.state.model.parameters()).device
    tokens = preprocess(q)
    ids = [app.state.vocab_to_int.get(tok, 0)
           for tok in tokens[:config.MAX_QRY_LEN]]

    # Pad to max length
    if len(ids) < config.MAX_QRY_LEN:
        ids += [0] * (config.MAX_QRY_LEN - len(ids))

    # Convert to tensor and get query embedding
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        query_emb = app.state.model.query_tower(x).cpu().numpy().tolist()[0]

    # Query ChromaDB
    results = app.state.collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )

    # Format results
    search_results = []
    for i in range(len(results['ids'][0])):
        search_results.append(
            SearchResult(
                id=results['ids'][0][i],
                document=results['documents'][0][i],
                similarity=float(results['distances'][0]
                                 [i]) if 'distances' in results else 0.0
            )
        )

    return SearchResponse(query=q, results=search_results)

# Run with: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
