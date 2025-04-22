import torch
import numpy as np
from typing import List, Tuple
from transformers import AutoTokenizer
import os
from tqdm import tqdm
from src.model import TwoTowerModel

class SearchEngine:
    def __init__(self, model_path: str, model_name: str = 'bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Initialize the model architecture
        self.model = TwoTowerModel(model_name=model_name)
        
        # Load the state dictionary
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        self.doc_encodings = None
        self.doc_ids = None

    def encode_query(self, query: str) -> torch.Tensor:
        """Encode a single query into its vector representation."""
        inputs = self.tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            query_repr, _ = self.model(inputs, inputs)  # We only need the query representation
        return query_repr.squeeze(0)  # Remove batch dimension

    def cache_document_encodings(self, documents: List[str], doc_ids: List[str], batch_size: int = 32):
        """Pre-cache document encodings for faster search."""
        self.doc_ids = doc_ids
        all_encodings = []
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Caching document encodings"):
            batch_docs = documents[i:i + batch_size]
            inputs = self.tokenizer(batch_docs, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                _, doc_reprs = self.model(inputs, inputs)  # We only need the document representations
            all_encodings.append(doc_reprs.cpu())
        
        self.doc_encodings = torch.cat(all_encodings, dim=0)
        print(f"Cached {len(self.doc_encodings)} document encodings")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for the most relevant documents for a given query."""
        if self.doc_encodings is None:
            raise ValueError("Document encodings not cached. Call cache_document_encodings first.")
        
        # Encode query
        query_repr = self.encode_query(query)
        
        # Move document encodings to the same device as query
        doc_encodings = self.doc_encodings.to(self.device)
        
        # Compute similarities
        similarities = torch.nn.functional.cosine_similarity(
            query_repr.unsqueeze(0),  # Add batch dimension
            doc_encodings,
            dim=1
        )
        
        # Get top k results
        top_k_indices = torch.topk(similarities, k=min(top_k, len(similarities))).indices
        results = [(self.doc_ids[idx], similarities[idx].item()) for idx in top_k_indices]
        
        return results

    def save_cache(self, cache_path: str):
        """Save cached document encodings to disk."""
        if self.doc_encodings is None:
            raise ValueError("No document encodings to save")
        
        torch.save({
            'encodings': self.doc_encodings,
            'doc_ids': self.doc_ids
        }, cache_path)
        print(f"Saved document encodings to {cache_path}")

    def load_cache(self, cache_path: str):
        """Load cached document encodings from disk."""
        cache = torch.load(cache_path, map_location=self.device)
        self.doc_encodings = cache['encodings']
        self.doc_ids = cache['doc_ids']
        print(f"Loaded {len(self.doc_encodings)} document encodings from {cache_path}") 