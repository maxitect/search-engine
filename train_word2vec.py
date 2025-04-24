import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from collections import Counter
import numpy as np
from tqdm import tqdm
import wandb
import os
import re
import math
import random
import time

class Word2VecDataset(Dataset):
    def __init__(self, words, word2idx, idx2word, window_size=5, min_count=5, subsample=1e-5):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.vocab_size = len(word2idx)
        self.window_size = window_size
        self.subsample = subsample
        
        # Calculate word frequencies for subsampling
        word_counts = Counter(words)
        total_words = sum(word_counts.values())
        self.word_freqs = {word: count/total_words for word, count in word_counts.items()}
        
        # Create training pairs with subsampling
        self.pairs = []
        for i in range(window_size, len(words)-window_size):
            target_word = words[i]
            
            # Apply subsampling
            if random.random() > self.subsample_prob(target_word, subsample):
                continue
                
            context_words = words[i-window_size:i] + words[i+1:i+window_size+1]
            if target_word in word2idx and all(c in word2idx for c in context_words):
                self.pairs.append((
                    torch.tensor([word2idx[c] for c in context_words], dtype=torch.long),
                    torch.tensor(word2idx[target_word], dtype=torch.long)
                ))
    
    def subsample_prob(self, word, threshold):
        freq = self.word_freqs.get(word, 0)
        return (math.sqrt(freq / threshold) + 1) * (threshold / freq)
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        return self.pairs[idx]

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, negative_samples=10):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.negative_samples = negative_samples
        
        # Initialize embeddings with uniform distribution
        init_range = 0.5 / embedding_dim
        self.embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, context_words, target_words):
        # Positive examples
        context_embeds = self.context_embeddings(context_words).mean(dim=1)
        target_embeds = self.embeddings(target_words)
        positive_score = torch.matmul(target_embeds, context_embeds.t()).diag().sigmoid().log()
        
        # Negative sampling
        noise = torch.randint(0, self.vocab_size, (target_words.shape[0], self.negative_samples), 
                            device=target_words.device)
        noise_embeds = self.embeddings(noise)
        negative_score = torch.bmm(noise_embeds, context_embeds.unsqueeze(2)).sigmoid().log().sum(1)
        
        return -(positive_score + negative_score).mean()
    
    def get_embeddings(self):
        return (self.embeddings.weight + self.context_embeddings.weight) / 2

def preprocess(text, min_count=5):
    """Improved text preprocessing"""
    text = text.lower()
    
    # Protect special tokens
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
    
    # Handle contractions
    text = re.sub(r"(\w+)'s", r"\1 's", text)
    text = re.sub(r"(\w+)n't", r"\1 n't", text)
    text = re.sub(r"(\w+)'re", r"\1 're", text)
    
    # Split and filter
    words = [w for w in text.split() 
             if len(w) > 1 and not w.isdigit() 
             and not any(c.isdigit() for c in w)]
    
    return words

def build_vocab(words, min_count=5):
    word_counts = Counter(words)
    vocab = [word for word, count in word_counts.items() if count >= min_count]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return vocab, word2idx, idx2word

def get_similar_words(embeddings, word, word2idx, idx2word, top_k=10):
    if word not in word2idx:
        return []
    
    word_vec = embeddings[word2idx[word]].unsqueeze(0)
    similarities = torch.cosine_similarity(word_vec, embeddings, dim=1)
    top_indices = torch.topk(similarities, k=top_k+1)[1][1:]  # Exclude self
    
    return [(idx2word[idx.item()], similarities[idx].item()) 
            for idx in top_indices]

def evaluate_analogies(embeddings, word2idx, idx2word, analogy_tests):
    correct = 0
    skipped = 0
    
    for a, b, c, expected in analogy_tests:
        if any(w not in word2idx for w in [a, b, c, expected]):
            skipped += 1
            continue
            
        vec_a = embeddings[word2idx[a]]
        vec_b = embeddings[word2idx[b]]
        vec_c = embeddings[word2idx[c]]
        
        target_vec = vec_b - vec_a + vec_c
        similarities = torch.cosine_similarity(target_vec.unsqueeze(0), embeddings, dim=1)
        
        # Exclude input words
        for w in [a, b, c]:
            similarities[word2idx[w]] = -1
            
        predicted = idx2word[torch.argmax(similarities).item()]
        if predicted == expected:
            correct += 1
    
    return correct / max(1, len(analogy_tests) - skipped), skipped

def train():
    wandb.init(project="word2vec-cbow-improved", config={
        "embedding_dim": 300,
        "window_size": 8,
        "batch_size": 2048,
        "gradient_accumulation_steps": 4,
        "min_count": 10,
        "initial_lr": 0.025,
        "min_lr": 0.0001,
        "epochs": 20,
        "negative_samples": 10,
        "subsample_threshold": 1e-5
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set memory optimization settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.cuda.empty_cache()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    with open("data/text8", "r") as f:
        text8_words = preprocess(f.read())
    
    msmarco = load_dataset("ms_marco", "v1.1", split="train")
    msmarco_words = []
    for example in tqdm(msmarco, desc="Processing MS-MARCO"):
        for passage in example['passages']['passage_text']:
            msmarco_words.extend(preprocess(passage))
    
    all_words = text8_words + msmarco_words
    vocab, word2idx, idx2word = build_vocab(all_words, min_count=wandb.config.min_count)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create dataset and dataloader
    dataset = Word2VecDataset(
        all_words, 
        word2idx, 
        idx2word,
        window_size=wandb.config.window_size,
        min_count=wandb.config.min_count,
        subsample=wandb.config.subsample_threshold
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=wandb.config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda x: (torch.stack([item[0] for item in x]), torch.stack([item[1] for item in x]))
    )
    
    # Initialize model
    model = CBOW(
        len(vocab),
        embedding_dim=wandb.config.embedding_dim,
        negative_samples=wandb.config.negative_samples
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=wandb.config.initial_lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=wandb.config.epochs,
        eta_min=wandb.config.min_lr
    )
    scaler = torch.amp.GradScaler('cuda')
    
    # Test words and analogies
    test_words = ["computer", "technology", "data", "learning", "system"]
    analogy_tests = [
        ("man", "king", "woman", "queen"),
        ("france", "paris", "germany", "berlin"),
        ("good", "best", "bad", "worst"),
        ("cat", "cats", "dog", "dogs"),
        ("walk", "walked", "run", "ran")
    ]
    
    # Training loop
    for epoch in range(wandb.config.epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        optimizer.zero_grad()
        
        for batch_idx, (context, target) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
            # Move tensors to device
            context = context.to(device)
            target = target.to(device)
            
            with torch.amp.autocast('cuda'):
                loss = model(context, target)
                # Scale loss for gradient accumulation
                loss = loss / wandb.config.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            # Step optimizer only after accumulating gradients
            if (batch_idx + 1) % wandb.config.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Clear cache periodically
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()
            
            total_loss += loss.item() * wandb.config.gradient_accumulation_steps
        
        # Handle remaining gradients
        if (batch_idx + 1) % wandb.config.gradient_accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        scheduler.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            embeddings = model.get_embeddings()
            
            # Calculate similarities
            similarity_results = {}
            for word in test_words:
                if word in word2idx:
                    similar = get_similar_words(embeddings, word, word2idx, idx2word)
                    similarity_results[word] = ", ".join(
                        f"{w} ({s:.3f})" for w, s in similar[:5]
                    )
            
            # Evaluate analogies
            analogy_acc, skipped = evaluate_analogies(
                embeddings, word2idx, idx2word, analogy_tests
            )
        
        # Log metrics
        wandb.log({
            "epoch": epoch + 1,
            "loss": total_loss / len(dataloader),
            "learning_rate": scheduler.get_last_lr()[0],
            "analogy_accuracy": analogy_acc,
            **{f"similar_to_{w}": s for w, s in similarity_results.items()}
        })
        
        # Print results
        print(f"\nEpoch {epoch+1} completed in {time.time()-start_time:.1f}s")
        print(f"Loss: {total_loss/len(dataloader):.4f}")
        print(f"Analogy accuracy: {analogy_acc*100:.1f}% (skipped {skipped})")
        for word, similar in similarity_results.items():
            print(f"Similar to '{word}': {similar}")
        
        # Clear cache after each epoch
        torch.cuda.empty_cache()
    
    # Save final embeddings
    final_embeddings = model.get_embeddings().cpu().numpy()
    os.makedirs("data/word2vec", exist_ok=True)
    np.save("data/word2vec/embeddings.npy", final_embeddings)
    with open("data/word2vec/vocab.txt", "w") as f:
        f.write("\n".join(vocab))
    
    wandb.finish()

if __name__ == "__main__":
    train()