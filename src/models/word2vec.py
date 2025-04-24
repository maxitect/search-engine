"""
Simple Word2Vec model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class Word2Vec(nn.Module):
    """
    Simple Word2Vec model with CBOW architecture.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Use sparse embeddings to save memory
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        
        # Use a smaller linear layer with bias
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=True)
        
        # Initialize weights with smaller values
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights with small random values."""
        init_range = 0.01  # Smaller initialization range
        self.embeddings.weight.data.uniform_(-init_range, init_range)
        self.linear.weight.data.uniform_(-init_range, init_range)
        self.linear.bias.data.zero_()
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            context (torch.Tensor): Context word indices of shape (batch_size, window_size*2)
        
        Returns:
            torch.Tensor: Log probabilities of shape (batch_size, vocab_size)
        """
        # Get embeddings for context words
        embeds = self.embeddings(context)  # (batch_size, window_size*2, embedding_dim)
        
        # Average the embeddings
        avg_embeds = embeds.mean(dim=1)  # (batch_size, embedding_dim)
        
        # Project to vocabulary size
        output = self.linear(avg_embeds)  # (batch_size, vocab_size)
        
        # Apply log softmax
        return F.log_softmax(output, dim=1)
    
    def find_similar_words(self, word_idx: int, word_to_idx: dict, idx_to_word: dict, 
                          top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find similar words using cosine similarity.
        
        Args:
            word_idx (int): Index of the word to find similar words for
            word_to_idx (dict): Word to index mapping
            idx_to_word (dict): Index to word mapping
            top_k (int): Number of similar words to return
        
        Returns:
            List[Tuple[str, float]]: List of (word, similarity) tuples
        """
        # Get embedding for the word
        word_embedding = self.embeddings.weight[word_idx].detach()
        
        # Calculate cosine similarities with all words
        similarities = F.cosine_similarity(
            word_embedding.unsqueeze(0),
            self.embeddings.weight,
            dim=1
        )
        
        # Get top k similar words (excluding the word itself)
        top_indices = torch.topk(similarities, top_k + 1)[1][1:]  # Skip the word itself
        similar_words = [
            (idx_to_word[idx.item()], similarities[idx].item())
            for idx in top_indices
        ]
        
        return similar_words 