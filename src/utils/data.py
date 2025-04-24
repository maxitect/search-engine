"""
Data processing utilities for Word2Vec training.
"""

from collections import Counter
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset, DataLoader

class Word2VecDataset(Dataset):
    """
    Dataset class for Word2Vec training.
    """
    
    def __init__(self, words: List[str], word_to_idx: Dict[str, int], window_size: int):
        self.words = words
        self.word_to_idx = word_to_idx
        self.window_size = window_size
    
    def __len__(self) -> int:
        return len(self.words) - 2 * self.window_size
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get target word
        target_idx = idx + self.window_size
        target = self.word_to_idx[self.words[target_idx]]
        
        # Get context words
        start = target_idx - self.window_size
        end = target_idx + self.window_size + 1
        context = [self.word_to_idx[self.words[i]] 
                  for i in range(start, end) if i != target_idx]
        
        return torch.tensor(context), torch.tensor(target)

def load_text8_data(file_path: str) -> List[str]:
    """
    Load and preprocess text8 data.
    
    Args:
        file_path (str): Path to text8 file
    
    Returns:
        List[str]: List of words
    """
    print("Loading text8 data...")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    return text.split()

def build_vocabulary(words: List[str], min_freq: int = 5, max_vocab_size: int = None) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    """
    Build vocabulary from words.
    
    Args:
        words (List[str]): List of words
        min_freq (int): Minimum word frequency
        max_vocab_size (int, optional): Maximum vocabulary size
    
    Returns:
        Tuple[List[str], Dict[str, int], Dict[int, str]]: 
            - List of vocabulary words
            - Word to index mapping
            - Index to word mapping
    """
    print("Building vocabulary...")
    
    # Count word frequencies
    word_counts = Counter(words)
    
    # Filter by minimum frequency
    vocab = [word for word, count in word_counts.items() if count >= min_freq]
    
    # Sort by frequency and limit vocabulary size
    vocab = sorted(vocab, key=lambda x: word_counts[x], reverse=True)
    if max_vocab_size is not None:
        vocab = vocab[:max_vocab_size]
    
    # Create mappings
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Most common words: {vocab[:10]}")
    return vocab, word_to_idx, idx_to_word

def create_data_loader(words: List[str], word_to_idx: Dict[str, int], 
                      window_size: int, batch_size: int) -> DataLoader:
    """
    Create DataLoader for training.
    
    Args:
        words (List[str]): List of words
        word_to_idx (Dict[str, int]): Word to index mapping
        window_size (int): Context window size
        batch_size (int): Batch size
    
    Returns:
        DataLoader: DataLoader for training
    """
    dataset = Word2VecDataset(words, word_to_idx, window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True) 