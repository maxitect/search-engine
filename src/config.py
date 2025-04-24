"""
Configuration settings for Word2Vec training.
"""

import os
from dataclasses import dataclass

@dataclass
class Config:
    # Data parameters
    window_size: int = 5
    min_word_freq: int = 50
    max_vocab_size: int = 5000
    
    # Model parameters
    embedding_dim: int = 8
    learning_rate: float = 0.001
    initial_batch_size: int = 64
    epochs: int = 5
    
    # Training optimizations
    use_sparse: bool = True
    use_mixed_precision: bool = True
    gradient_accumulation_steps: int = 256
    chunk_size: int = 50000
    memory_safety_factor: float = 0.3
    
    # Paths
    text8_path: str = "data/text8"
    output_dir: str = "data/word2vec"
    
    # Evaluation
    test_words: list = ("computer", "technology", "data", "learning", "system")
    eval_interval: int = 1000
    top_k: int = 10
    
    def __post_init__(self):
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True) 