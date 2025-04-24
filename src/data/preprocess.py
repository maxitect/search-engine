"""
Data preprocessing module for Word2Vec training.
"""

import os
import re
from collections import Counter
from typing import Tuple, Dict, List

from ..config import Config

def preprocess_text(text: str) -> str:
    """Preprocess text by converting to lowercase and replacing punctuation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data() -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """Load and preprocess data from text8 dataset."""
    print("Loading and preprocessing data...")
    
    words = []
    try:
        with open(Config.text8_path, 'r', encoding='utf-8') as f:
            chunk_size = 10000000  # Process 10M characters at a time
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                
                chunk = preprocess_text(chunk)
                words.extend(chunk.split())
                
                print(f"Processed {len(words)} words...")
        
        print(f"Loaded text8 dataset: {len(words)} words")
    except Exception as e:
        print(f"Error loading text8 dataset: {e}")
        words = []
    
    # Count word frequencies
    print("Counting word frequencies...")
    word_counts = Counter(words)
    
    # Filter by frequency and limit vocabulary size
    print("Filtering vocabulary...")
    most_common_words = [word for word, count in word_counts.most_common(Config.max_vocab_size) 
                         if count >= Config.min_word_freq]
    
    # Create word-to-index and index-to-word mappings
    word_to_idx = {word: idx for idx, word in enumerate(most_common_words)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # Filter words to only include vocabulary words
    filtered_words = [word for word in words if word in word_to_idx]
    
    print(f"Vocabulary size: {len(word_to_idx)}")
    print(f"Total words after filtering: {len(filtered_words)}")
    
    return filtered_words, word_to_idx, idx_to_word 