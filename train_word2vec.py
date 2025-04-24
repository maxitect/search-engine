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
    vocab_min_count = 100  # Increased from 50 to reduce vocabulary size
    
    # Model architecture
    embedding_dim = 300    # Reduced from 200 to save memory
    window_size = 5        # Context window size
    negative_samples = 5   # Number of negative samples
    
    # Training
    batch_size = 2048       # Significantly reduced to prevent OOM 512
    grad_accumulation = 8  # Accumulate gradients to simulate larger batch
    initial_lr = 0.025
    min_lr = 0.0001
    epochs = 10
    use_mixed_precision = True  # Enable mixed precision training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data processing
    max_words = 200000000   # Limit number of words to process (set to None to use all)
    use_msmarco = True     # Set to False to train only on text8 data

def load_text8():
    """Load and preprocess the text8 dataset"""
    print("Loading text8 dataset...")
    with open(Config.text8_path, "r") as f:
        text = f.read()
    return preprocess(text)

def load_msmarco():
    """Load and preprocess MS-MARCO dataset"""
    if not Config.use_msmarco:
        print("Skipping MS-MARCO dataset")
        return []
        
    print("Loading MS-MARCO dataset...")
    dataset = load_dataset("ms_marco", "v1.1", split="train")
    words = []
    for example in tqdm(dataset, desc="Processing MS-MARCO"):
        for passage in example['passages']['passage_text']:
            words.extend(preprocess(passage))
            # Check if we've reached the word limit
            if Config.max_words and len(words) >= Config.max_words // 2:
                print(f"Reached word limit for MS-MARCO: {len(words):,}")
                return words
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
        context, target = self.pairs[idx]
        
        # Convert to torch tensors
        context_tensor = torch.tensor(context, dtype=torch.long)
        target_tensor = torch.tensor(target, dtype=torch.long)
        
        return context_tensor, target_tensor

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
        pos_score = torch.sum(target_vec * context_vec, dim=1).sigmoid().log()
      
        # Negative sampling
        noise = torch.randint(0, self.vocab_size, 
                            (target.shape[0], self.negative_samples),
                            device=target.device)
        noise_vec = self.embeddings(noise)
        neg_score = torch.bmm(noise_vec, context_vec.unsqueeze(2)).squeeze().sigmoid().log().sum(1)
        
        return -(pos_score + neg_score).mean()
    
    def get_embeddings(self):
        return (self.embeddings.weight + self.context_embeddings.weight) / 2

def train():
    # Create output directory
    os.makedirs(Config.output_dir, exist_ok=True)
    
    # Load and combine datasets
    print("Loading and preprocessing data...")
    text8_words = load_text8()
    if Config.max_words:
        text8_words = text8_words[:Config.max_words // 2]
        print(f"Limited text8 words to {len(text8_words):,}")
    
    msmarco_words = load_msmarco()
    combined_words = text8_words + msmarco_words
    
    print(f"\nDataset stats:")
    print(f"- text8 words: {len(text8_words):,}")
    print(f"- MS-MARCO words: {len(msmarco_words):,}")
    print(f"- Total words: {len(combined_words):,}")
    
    # Build vocabulary
    vocab, word2idx, idx2word = build_vocab(combined_words, Config.vocab_min_count)
    print(f"\nVocabulary size: {len(vocab):,} (min count={Config.vocab_min_count})")
    
    # Free up memory
    del combined_words
    
    # Create dataset and dataloader
    dataset = Word2VecDataset(text8_words + msmarco_words, word2idx, Config.window_size)
    dataloader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=2,  # Reduced from 4
        pin_memory=True
    )
    
    # Free up more memory
    del text8_words
    del msmarco_words
    
    # Initialize model
    model = CBOW(len(vocab), Config.embedding_dim, Config.negative_samples)
    try:
        model = model.to(Config.device)
    except RuntimeError:
        print("Failed to move model to GPU, falling back to CPU")
        Config.device = "cpu"
        model = model.to(Config.device)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.initial_lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=Config.epochs * len(dataloader) // Config.grad_accumulation, 
        eta_min=Config.min_lr
    )
    
    # Set up mixed precision training if enabled and on GPU
    scaler = torch.cuda.amp.GradScaler() if Config.use_mixed_precision and Config.device == "cuda" else None
    
    # Training loop
    best_loss = float('inf')
    print(f"\nStarting training on {Config.device.upper()}...")
    print(f"Using mixed precision: {Config.use_mixed_precision and Config.device == 'cuda'}")
    print(f"Gradient accumulation steps: {Config.grad_accumulation}")
    
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        # Use tqdm for progress tracking
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.epochs}")
        
        for i, (context, target) in enumerate(progress_bar):
            context = context.to(Config.device, non_blocking=True)
            target = target.to(Config.device, non_blocking=True)
            
            # Mixed precision forward pass
            if scaler:
                with torch.cuda.amp.autocast():
                    loss = model(context, target) / Config.grad_accumulation
                scaler.scale(loss).backward()
            else:
                loss = model(context, target) / Config.grad_accumulation
                loss.backward()
            
            # Update metrics
            total_loss += loss.item() * Config.grad_accumulation
            
            # Update parameters after accumulating gradients
            if (i + 1) % Config.grad_accumulation == 0 or (i + 1) == len(dataloader):
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                    
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                
                # Update progress bar
                progress_bar.set_postfix({"loss": total_loss / (i + 1), "lr": scheduler.get_last_lr()[0]})
        
        avg_loss = total_loss / len(dataloader)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            
            # Save model to CPU to avoid GPU memory issues
            model_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
            
            torch.save({
                'model_state_dict': model_cpu,
                'vocab': vocab,
                'word2idx': word2idx,
                'idx2word': idx2word,
                'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('__')}
            }, os.path.join(Config.output_dir, "best_model.pt"))
            
            # Save embeddings in chunks to avoid memory issues
            embeddings = model.get_embeddings().cpu().detach().numpy()
            np.save(os.path.join(Config.output_dir, "embeddings.npy"), embeddings)
            del embeddings
            
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {scheduler.get_last_lr()[0]:.6f}")

if __name__ == "__main__":
    train()