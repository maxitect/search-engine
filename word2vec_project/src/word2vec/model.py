import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import numpy as np

class Word2Vec(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size,)
        embeds = self.embeddings(x)  # shape: (batch_size, embedding_dim)
        out = self.linear(embeds)    # shape: (batch_size, vocab_size)
        return out
    
    def get_word_vector(self, word_idx: int) -> torch.Tensor:
        """Get the embedding vector for a word."""
        return self.embeddings(torch.tensor([word_idx]))
    
    def get_similar_words(self, word_idx: int, top_k: int = 5) -> List[Tuple[int, float]]:
        """Get the most similar words to a given word."""
        word_vec = self.get_word_vector(word_idx)
        similarities = torch.matmul(self.embeddings.weight, word_vec.squeeze())
        top_k_values, top_k_indices = torch.topk(similarities, k=top_k)
        return list(zip(top_k_indices.tolist(), top_k_values.tolist()))
    
    def save_embeddings(self, path: str):
        """Save word embeddings to a file."""
        embeddings = self.embeddings.weight.detach().numpy()
        np.save(path, embeddings)
        print(f"Embeddings saved to {path}")
    
    def load_embeddings(self, path: str):
        """Load word embeddings from a file."""
        embeddings = np.load(path)
        self.embeddings.weight.data = torch.from_numpy(embeddings)
        print(f"Embeddings loaded from {path}") 