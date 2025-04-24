import torch
import torch.nn as nn
import torch.nn.functional as F
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
    vocab_min_count = 100  # Only keep frequent words
    
    # Model architecture
    embedding_dim = 300    # Embedding dimension
    window_size = 5        # Context window size
    negative_samples = 15  # Number of negative samples
    
    # Training
    batch_size = 4096      # Reduced batch size for better gradient quality
    grad_accumulation = 4  # Gradient accumulation steps
    initial_lr = 0.05      # Increased learning rate
    min_lr = 0.0001        # Minimum learning rate (should be lower than initial)
    epochs = 20            # Increased epochs for better convergence
    patience = 3           # Early stopping patience
    use_mixed_precision = True
    weight_decay = 1e-5    # Small weight decay
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data processing
    max_words = None       # No word limit
    use_msmarco = True     # Whether to use MS-MARCO

def load_text8():
    """Load full text8 dataset without limits"""
    print("Loading text8 dataset...")
    with open(Config.text8_path, "r") as f:
        text = f.read()
    return preprocess(text)

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
    """Build vocabulary with frequency information"""
    word_counts = Counter(words)
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    
    # Create frequency distribution for negative sampling
    word_freq = np.array([word_counts[word] for word in vocab], dtype=np.float32)
    word_freq /= word_freq.sum()
    
    return vocab, word2idx, idx2word, word_freq

class Word2VecDataset(Dataset):
    def __init__(self, words, word2idx, window_size):
        self.word2idx = word2idx
        self.window_size = window_size
        self.word_counts = Counter(words)
        self.total_words = sum(self.word_counts.values())
        self.pairs = self._generate_pairs(words)
        
    def _subsample_prob(self, word):
        """Calculate subsampling probability for discarding frequent words"""
        freq = self.word_counts[word] / self.total_words
        threshold = 1e-5
        return 1.0 - math.sqrt(threshold / freq) if freq > threshold else 1.0
        
    def _generate_pairs(self, words):
        """Generate training pairs with subsampling"""
        pairs = []
        for i in range(self.window_size, len(words)-self.window_size):
            target = words[i]
            if target not in self.word2idx:
                continue
                
            # Decide whether to keep this word based on frequency
            if random.random() > self._subsample_prob(target):
                continue
                
            # Get context words
            context = words[i-self.window_size:i] + words[i+1:i+self.window_size+1]
            valid_context = [c for c in context if c in self.word2idx]
            
            if len(valid_context) > 0:
                pairs.append(([self.word2idx[c] for c in valid_context], self.word2idx[target]))
                
        print(f"Generated {len(pairs):,} training pairs")
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        context, target = self.pairs[idx]
        # Ensure consistent context length by padding or truncating
        context = context[:2*self.window_size]  # Truncate if too long
        context = context + [0] * (2*self.window_size - len(context))  # Pad if too short
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long)

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim, negative_samples, word_freq):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        self.register_buffer('word_freq', torch.from_numpy(word_freq).float())
        self._init_weights()
    
    def _init_weights(self):
        """Xavier/Glorot initialization"""
        nn.init.xavier_uniform_(self.embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)
    
    def forward(self, context, target):
        batch_size = context.size(0)
        
        # Context vector (average of context words)
        context_vec = self.context_embeddings(context).mean(dim=1)
        
        # Positive score
        target_vec = self.embeddings(target)
        pos_score = torch.bmm(target_vec.unsqueeze(1), context_vec.unsqueeze(2)).squeeze()
        pos_loss = F.logsigmoid(pos_score)
        
        # Negative sampling with frequency-based distribution
        noise_words = torch.multinomial(
            self.word_freq.pow(0.75),
            batch_size * self.negative_samples,
            replacement=True
        ).view(batch_size, self.negative_samples).to(context.device)
        
        # Negative score
        noise_vec = self.embeddings(noise_words)
        neg_score = torch.bmm(noise_vec, context_vec.unsqueeze(2)).squeeze()
        neg_loss = F.logsigmoid(-neg_score).sum(1)
        
        return -(pos_loss + neg_loss).mean()
    
    def get_word_embeddings(self):
        """Get input embeddings for similarity tasks"""
        return self.embeddings.weight
        
    def get_context_embeddings(self):
        """Get context embeddings"""
        return self.context_embeddings.weight
    
    def get_embeddings(self):
        """Average of input and output embeddings"""
        return (self.embeddings.weight + self.context_embeddings.weight) / 2

def evaluate_analogies(model, word2idx, idx2word):
    """Evaluate on word analogies (king - man + woman = queen)"""
    analogies = [
        ("king", "man", "woman", "queen"),
        ("paris", "france", "italy", "rome"),
        ("quick", "quickly", "slow", "slowly"),
        ("man", "woman", "boy", "girl"),
        ("good", "better", "bad", "worse"),
        ("buy", "bought", "sell", "sold")
    ]
    
    results = []
    for a, b, c, expected in analogies:
        if a in word2idx and b in word2idx and c in word2idx:
            a_vec = model.get_embeddings()[word2idx[a]]
            b_vec = model.get_embeddings()[word2idx[b]]
            c_vec = model.get_embeddings()[word2idx[c]]
            
            # a - b + c should be close to expected
            result_vec = a_vec - b_vec + c_vec
            
            # Find closest word
            sims = torch.cosine_similarity(result_vec.unsqueeze(0), model.get_embeddings(), dim=1)
            top_k = torch.topk(sims, k=5)[1]
            closest_words = [idx2word[idx.item()] for idx in top_k if idx.item() != word2idx[a] and idx.item() != word2idx[b] and idx.item() != word2idx[c]]
            results.append((a, b, c, expected, closest_words[:3]))
    
    # Print results
    print("\nAnalogy evaluation:")
    correct = 0
    for a, b, c, expected, predictions in results:
        is_correct = expected in predictions
        if is_correct:
            correct += 1
        print(f"{a} - {b} + {c} = {expected}: {predictions} {'✓' if is_correct else '✗'}")
    
    if results:
        print(f"Accuracy: {correct/len(results):.2f}")
    
    return results

def train():
    # Setup
    os.makedirs(Config.output_dir, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Load data
    print("Loading and preprocessing data...")
    text8_words = load_text8()
    msmarco_words = load_msmarco()
    
    print(f"\nDataset stats:")
    print(f"- text8 words: {len(text8_words):,}")
    print(f"- MS-MARCO words: {len(msmarco_words):,}")
    print(f"- Total words: {len(text8_words + msmarco_words):,}")
    
    # Build vocabulary with frequencies
    vocab, word2idx, idx2word, word_freq = build_vocab(text8_words + msmarco_words, Config.vocab_min_count)
    print(f"Vocabulary size: {len(vocab):,} (min count={Config.vocab_min_count})")
    
    # Create dataset and dataloader
    dataset = Word2VecDataset(text8_words + msmarco_words, word2idx, Config.window_size)
    dataloader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model with word frequencies
    model = CBOW(len(vocab), Config.embedding_dim, Config.negative_samples, word_freq)
    model = model.to(Config.device)
    
    # Use SGD optimizer (better for word2vec)
    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.initial_lr,
        weight_decay=Config.weight_decay
    )
    
    # Cosine learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(dataloader)*Config.epochs,
        eta_min=Config.min_lr
    )
    
    scaler = torch.amp.GradScaler(enabled=Config.use_mixed_precision and Config.device == "cuda")
    
    # Training loop
    best_loss = float('inf')
    epochs_without_improvement = 0
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            
            # Update metrics
            total_loss += loss.item() * Config.grad_accumulation
            progress_bar.set_postfix({
                "loss": total_loss / (i + 1),
                "lr": optimizer.param_groups[0]['lr']
            })
        
        avg_loss = total_loss / len(dataloader)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': vocab,
                'word2idx': word2idx,
                'idx2word': idx2word,
                'config': {k: v for k, v in Config.__dict__.items() if not k.startswith('__')}
            }, os.path.join(Config.output_dir, "best_model.pt"))
            
            # Save embeddings
            for emb_type in ["input", "context", "combined"]:
                if emb_type == "input":
                    embeddings = model.get_word_embeddings().cpu().detach().numpy()
                    np.save(os.path.join(Config.output_dir, "word_embeddings.npy"), embeddings)
                elif emb_type == "context":
                    embeddings = model.get_context_embeddings().cpu().detach().numpy()
                    np.save(os.path.join(Config.output_dir, "context_embeddings.npy"), embeddings)
                else:
                    embeddings = model.get_embeddings().cpu().detach().numpy()
                    np.save(os.path.join(Config.output_dir, "combined_embeddings.npy"), embeddings)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= Config.patience:
                print(f"No improvement for {Config.patience} epochs. Stopping early.")
                break
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, LR = {optimizer.param_groups[0]['lr']:.6f}")
        
        # Validation checks - do this every epoch
        with torch.no_grad():
            # Word similarities using input embeddings (usually better for similarity)
            test_words = ["computer", "data", "learning", "algorithm", "network"]
            print("\nWord similarities (input embeddings):")
            for word in test_words:
                if word in word2idx:
                    vec = model.get_word_embeddings()[word2idx[word]]
                    sims = torch.cosine_similarity(vec.unsqueeze(0), model.get_word_embeddings(), dim=1)
                    top_k = torch.topk(sims, k=6)[1][1:]  # Exclude self
                    similar_words = [idx2word[idx.item()] for idx in top_k]
                    print(f"'{word}': {similar_words}")
                else:
                    print(f"'{word}' not in vocabulary.")
            
            # Word analogies using combined embeddings
            if epoch % 5 == 0 or epoch == Config.epochs - 1:
                evaluate_analogies(model, word2idx, idx2word)
                
    print("\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    
    # Final evaluation
    print("\nFinal word similarities (input embeddings):")
    with torch.no_grad():
        model.eval()
        test_words = ["computer", "data", "learning", "algorithm", "network", 
                    "king", "woman", "city", "money", "food"]
        for word in test_words:
            if word in word2idx:
                vec = model.get_word_embeddings()[word2idx[word]]
                sims = torch.cosine_similarity(vec.unsqueeze(0), model.get_word_embeddings(), dim=1)
                top_k = torch.topk(sims, k=6)[1][1:]  # Exclude self
                similar_words = [idx2word[idx.item()] for idx in top_k]
                print(f"'{word}': {similar_words}")
            else:
                print(f"'{word}' not in vocabulary.")
                
        # Final analogy evaluation
        evaluate_analogies(model, word2idx, idx2word)

if __name__ == "__main__":
    train()