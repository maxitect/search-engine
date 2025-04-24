import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
import numpy as np
import os
import re
import math
import random
from tqdm import tqdm

# Configuration
class Config:
    # Data paths
    data_dir = "data"
    text8_path = os.path.join(data_dir, "text8")
    output_dir = os.path.join(data_dir, "word2vec")
    
    # Vocabulary
    vocab_min_count = 50  # Only keep words appearing ≥50 times
    
    # Model architecture
    embedding_dim = 200   # Reduced from 300 to save memory
    window_size = 5       # Context window size
    negative_samples = 5  # Number of negative samples
    
    # Training
    batch_size = 4096     # Reduced from 65536 to prevent OOM
    initial_lr = 0.025
    min_lr = 0.0001
    epochs = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"

def load_text8():
    """Load and preprocess the text8 dataset"""
    print("Loading text8 dataset...")
    with open(Config.text8_path, "r") as f:
        text = f.read()
    return preprocess(text)

def load_msmarco():
    """Load and preprocess MS-MARCO dataset"""
    print("Loading MS-MARCO dataset...")
    dataset = load_dataset("ms_marco", "v1.1", split="train")
    words = []
    for example in tqdm(dataset, desc="Processing MS-MARCO"):
        for passage in example['passages']['passage_text']:
            words.extend(preprocess(passage))
    return words

def preprocess(text):
    """Clean and tokenize text"""
    text = text.lower()
    
    # Replace special tokens
    special_tokens = {
        '.': ' <PERIOD> ',
        ',': ' <COMMA> ',
        '"': ' <QUOTE> ',
        ';': ' <SEMICOLON> ',
        '!': ' <EXCLAMATION> ',
        '?': ' <QUESTION> ',
        '(': ' <LPAREN> ',
        ')': ' <RPAREN> ',
        '--': ' <HYPHEN> ',
        ':': ' <COLON> ',
        "'": ' <APOSTROPHE> ',
        "\n": ' <NEWLINE> '
    }
    
    for k, v in special_tokens.items():
        text = text.replace(k, v)
    
    # Remove remaining special chars
    text = re.sub(r'[^a-z0-9<>\s]', ' ', text)
    
    # Tokenize and filter
    words = [w for w in text.split() 
             if len(w) > 1 and not w.isdigit() 
             and not any(c.isdigit() for c in w)]
    
    return words

def build_vocab(words, min_count):
    """Build vocabulary from words"""
    word_counts = Counter(words)
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return vocab, word2idx, idx2word

class Word2VecDataset(Dataset):
    def __init__(self, words, word2idx, window_size):
        self.word2idx = word2idx
        self.window_size = window_size
        self.word_counts = Counter(words)
        self.total_words = sum(self.word_counts.values())
        self.pairs = self._generate_pairs(words)
        
    def _subsample_prob(self, word):
        """Calculate subsampling probability"""
        freq = self.word_counts[word] / self.total_words
        threshold = 1e-5
        return (math.sqrt(freq / threshold) + 1) * (threshold / freq)
        
    def _generate_pairs(self, words):
        """Generate training pairs with subsampling"""
        pairs = []
        for i in range(self.window_size, len(words)-self.window_size):
            target = words[i]
            
            # Apply subsampling
            if random.random() > self._subsample_prob(target):
                continue
                
            context = words[i-self.window_size:i] + words[i+1:i+self.window_size+1]
            if target in self.word2idx and all(c in self.word2idx for c in context):
                pairs.append((
                    [self.word2idx[c] for c in context],
                    self.word2idx[target]
                ))
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, negative_samples):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        
        # Initialize embeddings
        init_range = 0.5 / embedding_dim
        self.embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, context, target):
        # Positive examples
        context_vec = self.context_embeddings(context).mean(dim=1)
        target_vec = self.embeddings(target)
        pos_score = torch.matmul(target_vec, context_vec.t()).diag().sigmoid().log()
        
        # Negative sampling
        noise = torch.randint(0, self.vocab_size, 
                            (target.shape[0], self.negative_samples),
                            device=target.device)
        noise_vec = self.embeddings(noise)
        neg_score = torch.bmm(noise_vec, context_vec.unsqueeze(2)).sigmoid().log().sum(1)
        
        return -(pos_score + neg_score).mean()
    
    def get_embeddings(self):
        return (self.embeddings.weight + self.context_embeddings.weight) / 2

def train():
    # Create output directory
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Load and combine datasets
    print("Loading and preprocessing data...")
    text8_words = load_text8()
    msmarco_words = load_msmarco()
    combined_words = text8_words + msmarco_words
    
    print(f"\nDataset stats:")
    print(f"- text8 words: {len(text8_words):,}")
    print(f"- MS-MARCO words: {len(msmarco_words):,}")
    print(f"- Total words: {len(combined_words):,}")
    
    # Build vocabulary
    vocab, word2idx, idx2word = build_vocab(combined_words, Config.vocab_min_count)
    print(f"\nVocabulary size: {len(vocab):,} (min count={Config.vocab_min_count})")
    
    # Create dataset and dataloader
    dataset = Word2VecDataset(combined_words, word2idx, Config.window_size)
    dataloader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = CBOW(len(vocab), Config.embedding_dim, Config.negative_samples)
    model = model.to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, Config.epochs, Config.min_lr)
    
    # Training loop
    best_loss = float('inf')
    print("\nStarting training...")
    
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        
        for context, target in tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.epochs}"):
            context = context.to(Config.device, non_blocking=True)
            target = target.to(Config.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            loss = model(context, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step()
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'word2idx': word2idx,
                'idx2word': idx2word,
                'config': Config.__dict__
            }, os.path.join(Config.output_dir, "best_model.pt"))
            
            # Save embeddings
            embeddings = model.get_embeddings().cpu().detach().numpy()
            np.save(os.path.join(Config.output_dir, "embeddings.npy"), embeddings)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")

if __name__ == "__main__":
    train()