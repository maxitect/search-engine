"""
Configuration settings for Word2Vec training.
"""

from dataclasses import dataclass

@dataclass
class Config:
    # Data parameters
    window_size: int = 5          # Context window size
    min_word_freq: int = 5        # Minimum word frequency
    embedding_dim: int = 100      # Dimension of word embeddings
    
    # Training parameters
    batch_size: int = 128         # Batch size for training
    learning_rate: float = 0.001  # Learning rate
    epochs: int = 5               # Number of training epochs
    
    # Paths
    data_path: str = "data/text8"  # Path to text8 dataset
    output_dir: str = "output"     # Directory for saving outputs
    
    def __post_init__(self):
        # Create output directory if it doesn't exist
        import os
        os.makedirs(self.output_dir, exist_ok=True) 