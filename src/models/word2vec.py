"""
Word2Vec model implementation with memory optimization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
import numpy as np

from ..config import Config

class Word2VecModel(nn.Module):
    """Memory-efficient Word2Vec model with CBOW architecture."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, use_sparse: bool = True):
        super(Word2VecModel, self).__init__()
        
        # Split vocabulary into chunks
        self.chunk_size = 2000
        self.num_chunks = (vocab_size + self.chunk_size - 1) // self.chunk_size
        
        # Create embedding chunks
        self.embeddings = nn.ModuleList([
            nn.Embedding(min(self.chunk_size, vocab_size - i * self.chunk_size), 
                        embedding_dim, sparse=use_sparse)
            for i in range(self.num_chunks)
        ])
        
        # Linear layer with tied weights
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=False)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        for emb in self.embeddings:
            emb.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
    
    def get_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        """Get embeddings from appropriate chunk with memory optimization."""
        embeddings = []
        for i in range(self.num_chunks):
            mask = (indices >= i * self.chunk_size) & (indices < (i + 1) * self.chunk_size)
            if mask.any():
                chunk_indices = indices[mask] - i * self.chunk_size
                chunk_emb = self.embeddings[i](chunk_indices)
                embeddings.append((mask, chunk_emb))
        
        if not embeddings:
            return torch.zeros(len(indices), self.embeddings[0].embedding_dim, device=indices.device)
        
        result = torch.zeros(len(indices), self.embeddings[0].embedding_dim, device=indices.device)
        for mask, emb in embeddings:
            result[mask] = emb
        return result
    
    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        """Forward pass with gradient checkpointing."""
        if self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, contexts)
        return self._forward(contexts)
    
    def _forward(self, contexts: torch.Tensor) -> torch.Tensor:
        """Actual forward pass implementation."""
        # Get embeddings for context
        embeds = self.get_embedding(contexts)
        embeds = embeds.view(-1, contexts.size(1), self.embeddings[0].embedding_dim)
        embeds = embeds.mean(dim=1)
        
        # Get output scores
        output = self.linear(embeds)
        return F.log_softmax(output, dim=1)
    
    def find_similar_words(self, word: str, word_to_idx: Dict[str, int], 
                          idx_to_word: Dict[int, str], top_k: int = 10) -> List[tuple]:
        """Find most similar words using cosine similarity."""
        if word not in word_to_idx:
            return []
        
        # Get the word index
        word_idx = word_to_idx[word]
        
        # Get the word embedding
        word_vector = self.get_embedding(torch.tensor([word_idx], device=self.embeddings[0].weight.device))
        word_vector = word_vector.detach().cpu().numpy()
        
        # Calculate similarities for a subset of words
        similarities = []
        for i in range(self.num_chunks):
            chunk_embeddings = self.embeddings[i].weight.detach().cpu().numpy()
            chunk_similarities = np.dot(chunk_embeddings, word_vector.T).flatten()
            for j, sim in enumerate(chunk_similarities):
                word_idx = i * self.chunk_size + j
                if word_idx in idx_to_word:
                    similarities.append((idx_to_word[word_idx], sim))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[1:top_k+1]  # Exclude the word itself 