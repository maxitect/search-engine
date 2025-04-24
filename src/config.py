"""
Configuration settings for Word2Vec training.
"""

from dataclasses import dataclass

@dataclass
class Config:
    # Data parameters
    window_size: int = 3          # Reduced window size
    min_word_freq: int = 500      # Increased minimum word frequency
    embedding_dim: int = 2        # Further reduced embedding dimension
    max_vocab_size: int = 500     # Further reduced vocabulary size
    
    # Training parameters
    batch_size: int = 8           # Further reduced batch size
    learning_rate: float = 0.001  # Learning rate
    epochs: int = 5               # Number of training epochs
    gradient_accumulation_steps: int = 16  # Increased gradient accumulation
    
    # Paths
    data_path: str = "data/text8"  # Path to text8 dataset
    output_dir: str = "output"     # Directory for saving outputs
    
    def __post_init__(self):
        # Create output directory if it doesn't exist
        import os
        os.makedirs(self.output_dir, exist_ok=True) 