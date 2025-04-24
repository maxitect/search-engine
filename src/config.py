"""
Configuration settings for Word2Vec training.
"""

from dataclasses import dataclass

@dataclass
class Config:
    # Data parameters
    window_size: int = 5          # Context window size
    min_word_freq: int = 50       # Increased minimum word frequency to reduce vocabulary size
    embedding_dim: int = 50       # Reduced embedding dimension
    max_vocab_size: int = 10000   # Limit vocabulary size
    
    # Training parameters
    batch_size: int = 64          # Reduced batch size
    learning_rate: float = 0.001  # Learning rate
    epochs: int = 5               # Number of training epochs
    
    # Paths
    data_path: str = "data/text8"  # Path to text8 dataset
    output_dir: str = "output"     # Directory for saving outputs
    
    def __post_init__(self):
        # Create output directory if it doesn't exist
        import os
        os.makedirs(self.output_dir, exist_ok=True) 