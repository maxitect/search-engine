"""
Word2Vec model implementation with CBOW architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Word2Vec(nn.Module):
    """
    Word2Vec model with Continuous Bag of Words (CBOW) architecture.
    
    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of word embeddings
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(Word2Vec, self).__init__()
        
        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Linear layer for prediction
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights with uniform distribution."""
        self.embeddings.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.zero_()
    
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            context (torch.Tensor): Tensor of context word indices
                                  Shape: (batch_size, 2 * window_size)
        
        Returns:
            torch.Tensor: Log probabilities for target word prediction
                         Shape: (batch_size, vocab_size)
        """
        # Get embeddings for context words
        # Shape: (batch_size, 2 * window_size, embedding_dim)
        embeds = self.embeddings(context)
        
        # Average the embeddings
        # Shape: (batch_size, embedding_dim)
        embeds = embeds.mean(dim=1)
        
        # Get output scores
        # Shape: (batch_size, vocab_size)
        output = self.linear(embeds)
        
        # Apply log softmax for numerical stability
        return F.log_softmax(output, dim=1)
    
    def get_embedding(self, word_idx: int) -> torch.Tensor:
        """
        Get embedding for a specific word.
        
        Args:
            word_idx (int): Index of the word
        
        Returns:
            torch.Tensor: Word embedding
        """
        return self.embeddings(torch.tensor([word_idx]))
    
    def find_similar_words(self, word_idx: int, word_to_idx: dict, idx_to_word: dict, top_k: int = 10):
        """
        Find most similar words using cosine similarity.
        
        Args:
            word_idx (int): Index of the target word
            word_to_idx (dict): Word to index mapping
            idx_to_word (dict): Index to word mapping
            top_k (int): Number of similar words to return
        
        Returns:
            list: List of (word, similarity) tuples
        """
        # Get the word embedding
        word_vector = self.get_embedding(word_idx).squeeze()
        
        # Calculate similarities with all other words
        similarities = []
        for idx in range(len(word_to_idx)):
            if idx != word_idx:
                other_vector = self.get_embedding(idx).squeeze()
                sim = F.cosine_similarity(word_vector, other_vector, dim=0)
                similarities.append((idx_to_word[idx], sim.item()))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k] 