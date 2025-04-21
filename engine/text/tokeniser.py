from __future__ import annotations

import logging
import re
from collections import Counter

import pandas as pd
import torch

logger = logging.getLogger(__name__)


def get_wiki_text(fp):
    df = pd.read_parquet(fp, engine='fastparquet')
    df_string = df.text.values[0]
    return df_string


def preprocess_text(text):
    """
    Tokenizes and normalizes text.

    Args:
        text: The input string.

    Returns:
        A list of lowercase tokens.
    """
    # Convert to lowercase
    text = text.lower()
    # Replace punctuation with spaces (basic approach)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Split into words on spaces
    tokens = text.split()
    return tokens


def build_vocab(tokens, min_freq=5):
    """
    Builds a vocabulary from a list of tokens.

    Args:
        tokens: A list of tokens (output from preprocess_text).
        min_freq: The minimum frequency for a word to be included in the vocabulary.

    Returns:
        A tuple containing:
            word_to_idx: Dictionary mapping words to indices.
            idx_to_word: List mapping indices to words.
            filtered_counts (list): Counts for the filtered words.
    """
    logger.info(f'Building vocabulary with min_freq={min_freq}')
    word_counts = Counter(tokens)
    # Filter words by minimum frequency
    filtered_words = []
    filtered_counts = []
    num_discarded = 0
    total_words = len(tokens)
    for word, count in word_counts.items():
        if count >= min_freq:
            num_discarded += count
            filtered_words.append(word)
            filtered_counts.append(count)

    # Sort by words for consistent indexing
    filtered_words, filtered_counts = zip(
        *sorted(
            zip(filtered_words, filtered_counts),
            key=lambda pair: pair[0],
        ),
    )
    filtered_words = list(filtered_words)
    filtered_counts = list(filtered_counts)

    # Create mappings
    # Often reserve index 0 for padding or unknown tokens
    word_to_idx = {word: i + 1 for i, word in enumerate(filtered_words)}
    idx_to_word = {i + 1: word for i, word in enumerate(filtered_words)}

    # Add an <UNK> token for unknown words
    word_to_idx['<UNK>'] = 0
    idx_to_word[0] = '<UNK>'
    filtered_counts.insert(0, num_discarded)
    return word_to_idx, idx_to_word, filtered_counts


class Vocabulary:
    def __init__(self, tokens_list: list, min_freq: int = 5):
        word_to_idx, idx_to_word, filtered_counts = build_vocab(
            tokens_list, min_freq=min_freq,
        )
        self.word_to_idx_dict = word_to_idx
        self.idx_to_word_list = idx_to_word
        self.filtered_counts = torch.tensor(filtered_counts)
        self.normalised_counts = self.filtered_counts/self.filtered_counts.sum()

    def __len__(self):
        return len(self.word_to_idx_dict)


class Tokeniser:
    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab
        self.word_to_idx_dict = vocab.word_to_idx_dict
        self.idx_to_word_list = vocab.idx_to_word_list
        self.vocab_size = len(vocab)

    def word_to_idx(self, word: str):
        if type(word) == str:
            try:
                return self.word_to_idx_dict[word]
            except KeyError:
                return self.word_to_idx_dict['<UNK>']
        else:
            raise ValueError(f'Word must be a string, got {type(word)}')

    def idx_to_word(self, idx: int):
        if type(idx) == int:
            try:
                return self.idx_to_word_list[idx]
            except KeyError:
                return self.word_to_idx_dict['<UNK>']
        else:
            raise ValueError(f'Index must be an integer, got {type(idx)}')

    def tokenise_list(self, text: list):
        return [self.word_to_idx(word) for word in text]

    def tokenise_string(self, text: str):
        return self.tokenise_list(preprocess_text(text))

    def tokens_to_words(self, tokens) -> list:
        if type(tokens) == torch.Tensor:
            if tokens.ndim == 1:
                return [self.idx_to_word(int(idx)) for idx in tokens]
            elif tokens.ndim == 0:
                return [self.idx_to_word(int(tokens))]
            else:
                raise ValueError(
                    f'Tokens must be a 0D or 1D tensor, got {tokens.ndim}D',
                )
        else:
            return [self.idx_to_word(int(idx)) for idx in tokens]


def get_tokeniser(fp: str = 'data/text8.parquet', min_freq: int = 5):
    text = get_wiki_text(fp)
    tokens = preprocess_text(text)
    vocab = Vocabulary(tokens, min_freq=min_freq)
    return Tokeniser(vocab)
