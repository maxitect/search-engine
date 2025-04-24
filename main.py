#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import wandb
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent))

from src.data.preprocess import load_and_preprocess_data
from src.models.word2vec import Word2VecModel
from src.training.trainer import train_word2vec
from src.utils.memory import get_gpu_memory
from src.utils.logging import setup_logging

def main():
    # Setup logging
    setup_logging()
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        total_memory = get_gpu_memory()
        print(f"Total GPU memory: {total_memory:.2f} MB")
    
    # Initialize wandb
    wandb.init(project="word2vec-training", name="cbow-memory-optimized")
    
    try:
        # Load and preprocess data
        print("Loading and preprocessing data...")
        filtered_words, word_to_idx, idx_to_word = load_and_preprocess_data()
        
        # Train model
        print("Starting training...")
        model, word_to_idx, idx_to_word = train_word2vec(
            filtered_words=filtered_words,
            word_to_idx=word_to_idx,
            idx_to_word=idx_to_word,
            device=device
        )
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise
    finally:
        # Complete wandb run
        wandb.finish()

if __name__ == "__main__":
    main()
