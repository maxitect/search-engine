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
    vocab_min_count = 5  # Only keep frequent words
    
    # Model architecture
    embedding_dim = 200    # Reduced from 300 for memory
    window_size = 5        # Context window size
    negative_samples = 5   # Number of negative samples
    
    # Training
    batch_size = 1024      # Reduced to prevent OOM
    grad_accumulation = 8  # Gradient accumulation steps
    initial_lr = 0.01      # Reduced learning rate
    min_lr = 0.0001
    epochs = 10            # Increased epochs
    use_mixed_precision = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data processing
    max_words = None   # Limit words to process
    use_msmarco = True     # Whether to use MS-MARCO

def load_text8():
    """Load full text8 dataset without limits"""
    print("Loading text8 dataset...")
    with open(Config.text8_path, "r") as f:
        text = f.read()
    return preprocess(text)  # Return all words without limiting

def load_msmarco():
    """Load full MS-MARCO dataset without limits"""
    if not Config.use_msmarco:
        return []
        
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
    special_tokens = {
        '.': ' <PERIOD> ', ',': ' <COMMA> ', '"': ' <QUOTE> ',
        ';': ' <SEMICOLON> ', '!': ' <EXCLAMATION> ', '?': ' <QUESTION> ',
        '(': ' <LPAREN> ', ')': ' <RPAREN> ', '--': ' <HYPHEN> ',
        ':': ' <COLON> ', "'": ' <APOSTROPHE> ', "\n": ' <NEWLINE> '
    }
    for k, v in special_tokens.items():
        text = text.replace(k, v)
    text = re.sub(r'[^a-z0-9<>\s]', ' ', text)
    return [w for w in text.split() if len(w) > 1 and not w.isdigit() and not any(c.isdigit() for c in w)]

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
            if random.random() > self._subsample_prob(target):
                continue
            context = words[i-self.window_size:i] + words[i+1:i+self.window_size+1]
            if target in self.word2idx and all(c in self.word2idx for c in context):
                pairs.append(([self.word2idx[c] for c in context], self.word2idx[target]))
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        context, target = self.pairs[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, negative_samples):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        self._init_weights()
    
    def _init_weights(self):
        """Proper initialization"""
        init_range = 0.5 / self.embeddings.embedding_dim
        nn.init.uniform_(self.embeddings.weight, -init_range, init_range)
        nn.init.uniform_(self.context_embeddings.weight, -init_range, init_range)
    
    def forward(self, context, target):
        # Positive examples
        context_vec = self.context_embeddings(context).mean(dim=1)
        target_vec = self.embeddings(target)
        
        # Positive score
        pos_score = torch.bmm(target_vec.unsqueeze(1), 
                            context_vec.unsqueeze(2)).squeeze().sigmoid().log()
        
        # Negative sampling
        noise = torch.randint(0, self.vocab_size,
                            (context.size(0), self.negative_samples),
                            device=context.device)
        noise_vec = self.embeddings(noise)
        neg_score = torch.bmm(noise_vec, context_vec.unsqueeze(2)).squeeze().sigmoid().log().sum(1)
        
        return -(pos_score + neg_score).mean()
    
    def get_embeddings(self):
        return (self.embeddings.weight + self.context_embeddings.weight) / 2

def train():
    # Setup
    os.makedirs(Config.output_dir, exist_ok=True)
    torch.manual_seed(42)
    
    # Load data
    print("Loading and preprocessing data...")
    text8_words = load_text8()
    msmarco_words = load_msmarco()
    
    print(f"\nDataset stats:")
    print(f"- text8 words: {len(text8_words):,}")
    print(f"- MS-MARCO words: {len(msmarco_words):,}")
    print(f"- Total words: {len(text8_words + msmarco_words):,}")
    
    # Build vocabulary
    vocab, word2idx, idx2word = build_vocab(text8_words + msmarco_words, Config.vocab_min_count)
    print(f"Vocabulary size: {len(vocab):,} (min count={Config.vocab_min_count})")
    
    # Create dataset and dataloader
    dataset = Word2VecDataset(text8_words + msmarco_words, word2idx, Config.window_size)
    dataloader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    model = CBOW(len(vocab), Config.embedding_dim, Config.negative_samples)
    try:
        model = model.to(Config.device)
    except RuntimeError:
        print("Failed to move model to GPU, falling back to CPU")
        Config.device = "cpu"
        model = model.to(Config.device)
    
    optimizer = optim.Adam(model.parameters(), lr=Config.initial_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=1,
        min_lr=Config.min_lr
    )
    
    scaler = torch.amp.GradScaler(init_scale=1024, enabled=Config.use_mixed_precision and Config.device == "cuda")
    
    # Training loop
    best_loss = float('inf')
    print(f"\nStarting training on {Config.device.upper()}...")
    print(f"Using mixed precision: {Config.use_mixed_precision and Config.device == 'cuda'}")
    print(f"Gradient accumulation steps: {Config.grad_accumulation}")
    
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.epochs}")
        
        for i, (context, target) in enumerate(progress_bar):
            context = context.to(Config.device, non_blocking=True)
            target = target.to(Config.device, non_blocking=True)
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type=Config.device, enabled=scaler is not None):
                loss = model(context, target) / Config.grad_accumulation
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (i + 1) % Config.grad_accumulation == 0 or (i + 1) == len(dataloader):
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            # Update metrics
            total_loss += loss.item() * Config.grad_accumulation
            progress_bar.set_postfix({"loss": total_loss / (i + 1), "lr": optimizer.param_groups[0]['lr']})
        
        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'word2idx': word2idx,
                'idx2word': idx2word,
                'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('__')}
            }, os.path.join(Config.output_dir, "best_model.pt"))
            
            # Save embeddings
            embeddings = model.get_embeddings().cpu().detach().numpy()
            np.save(os.path.join(Config.output_dir, "embeddings.npy"), embeddings)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # Validation checks
        if epoch % 5 == 0:
            with torch.no_grad():
                test_words = ["computer", "data", "learning"]
                for word in test_words:
                    if word in word2idx:
                        vec = model.get_embeddings()[word2idx[word]]
                        sims = torch.cosine_similarity(vec, model.get_embeddings(), dim=1)
                        top_k = torch.topk(sims, k=6)[1][1:]  # Exclude self
                         similar_words = [idx2word[idx] for idx in top_k if idx != word2idx[word]]
                        print(f"Top similar words to '{word}': {similar_words}")
                    else:
                        print(f"'{word}' not in vocabulary.")

if __name__ == "__main__":
    train()