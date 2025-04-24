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
from tqdm import tqdm

# Configuration - MODIFIED FOR MEMORY EFFICIENCY
class Config:
    # Data
    data_dir = "data"
    vocab_min_count = 50  # Increased to reduce vocabulary size
    subsample_threshold = 1e-5
    
    # Model - REDUCED DIMENSIONS
    embedding_dim = 200  # Reduced from 300
    window_size = 3  # Smaller context window
    negative_samples = 5  # Reduced negative samples
    
    # Training - SMALLER BATCHES
    batch_size = 2048  # Reduced from 5000
    initial_lr = 0.025
    min_lr = 0.0001
    epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

# Text preprocessing (unchanged)
def preprocess(text):
    text = text.lower()
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
    
    text = re.sub(r'[^a-z0-9<>\s]', ' ', text)
    words = [w for w in text.split() if len(w) > 1 and not w.isdigit()]
    return words

# MODIFIED Dataset class with subsampling
class Word2VecDataset(Dataset):
    def __init__(self, words, word2idx, window_size):
        self.word2idx = word2idx
        self.window_size = window_size
        self.word_counts = Counter(words)
        self.total_words = sum(self.word_counts.values())
        self.pairs = self._generate_pairs(words)
        
    def _subsample_prob(self, word):
        freq = self.word_counts[word] / self.total_words
        threshold = Config.subsample_threshold
        return (math.sqrt(freq / threshold) + 1) * (threshold / freq)
        
    def _generate_pairs(self, words):
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

# MODIFIED CBOW Model with memory optimizations
class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, negative_samples):
        super().__init__()
        # Use float16 to save memory
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, dtype=torch.float16)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim, dtype=torch.float16)
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        
        # Initialize embeddings
        init_range = 0.5 / embedding_dim
        self.embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, context, target):
        # Convert to float32 for calculation
        context = context.float()
        target = target.float()
        
        # Positive examples
        context_vec = self.context_embeddings(context.long()).mean(dim=1)
        target_vec = self.embeddings(target.long())
        pos_score = torch.matmul(target_vec, context_vec.t()).diag().sigmoid().log()
        
        # Negative sampling
        noise = torch.randint(0, self.vocab_size, 
                             (target.shape[0], self.negative_samples),
                             device=target.device)
        noise_vec = self.embeddings(noise.long())
        neg_score = torch.bmm(noise_vec, context_vec.unsqueeze(2)).sigmoid().log().sum(1)
        
        return -(pos_score + neg_score).mean()
    
    def get_embeddings(self):
        return ((self.embeddings.weight + self.context_embeddings.weight) / 2).float()

# MODIFIED Training function with memory management
def train():
    # Clear GPU cache
    torch.cuda.empty_cache()
    
    # Load and combine datasets
    print("Loading text8...")
    text8_words = load_text8()
    print(f"Loaded {len(text8_words)} words from text8")
    
    print("Loading MS-MARCO...")
    msmarco_words = load_msmarco()
    print(f"Loaded {len(msmarco_words)} words from MS-MARCO")
    
    combined_words = text8_words + msmarco_words
    print(f"Total words: {len(combined_words)}")
    
    # Build vocabulary with higher min_count
    vocab, word2idx, idx2word = build_vocab(combined_words, Config.vocab_min_count)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Free memory
    del text8_words, msmarco_words
    torch.cuda.empty_cache()
    
    # Create dataset and dataloader
    dataset = Word2VecDataset(combined_words, word2idx, Config.window_size)
    dataloader = DataLoader(dataset, 
                          batch_size=Config.batch_size, 
                          shuffle=True,
                          pin_memory=True)
    
    # Initialize model
    model = CBOW(len(vocab), Config.embedding_dim, Config.negative_samples)
    model = model.to(Config.device)
    optimizer = optim.Adam(model.parameters(), lr=Config.initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, Config.epochs, Config.min_lr)
    
    # Training loop
    best_loss = float('inf')
    os.makedirs(os.path.join(Config.data_dir, "word2vec"), exist_ok=True)
    
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        
        for context, target in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            context = context.to(Config.device, non_blocking=True)
            target = target.to(Config.device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            loss = model(context, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Periodic memory cleanup
            if len(dataloader) > 100 and (i+1) % 100 == 0:
                torch.cuda.empty_cache()
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save in float16 to save space
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'word2idx': word2idx,
                'idx2word': idx2word,
                'config': Config.__dict__
            }, os.path.join(Config.data_dir, "word2vec", "best_model.pt"))
            
            # Save embeddings as float16
            embeddings = model.get_embeddings().half().cpu().numpy()
            np.save(os.path.join(Config.data_dir, "word2vec", "embeddings.npy"), embeddings)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train()