#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import time
import math
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
import gc

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Enable cuDNN benchmark for better performance
torch.backends.cudnn.benchmark = True

# Initialize W&B
wandb.init(project="word2vec-training", name="cbow-sparse-embeddings")

class Config:
    # Data parameters
    window_size = 5
    min_word_freq = 5
    
    # Model parameters
    embedding_dim = 300
    learning_rate = 0.001
    batch_size = 2048
    epochs = 5
    
    # Training optimizations
    use_sparse = True
    use_mixed_precision = True
    gradient_accumulation_steps = 8  # Accumulate gradients over multiple batches
    chunks = 10  # Process vocabulary in chunks to save memory
    
    # Paths
    text8_path = "data/text8"
    msmarco_path = "data/msmarco-v1.1"
    output_dir = "data/word2vec"
    
    # Evaluation
    test_words = ["computer", "technology", "data", "learning", "system"]
    eval_interval = 5000  # Evaluate every n batches
    top_k = 10  # Number of similar words to find

# Create output directory if it doesn't exist
os.makedirs(Config.output_dir, exist_ok=True)

def preprocess_text(text):
    """Preprocess text by converting to lowercase and replacing punctuation with special tokens."""
    # Convert text to lowercase
    text = text.lower()
    
    # Replace punctuation with special tokens
    text = re.sub(r'\.', ' <PERIOD> ', text)
    text = re.sub(r',', ' <COMMA> ', text)
    text = re.sub(r'"', ' <QUOTATION> ', text)
    text = re.sub(r';', ' <SEMICOLON> ', text)
    text = re.sub(r'!', ' <EXCLAMATION> ', text)
    text = re.sub(r'\?', ' <QUESTION> ', text)
    text = re.sub(r'\(|\)', ' <PAREN> ', text)
    text = re.sub(r'--', ' <HYPHEN> ', text)
    text = re.sub(r':', ' <COLON> ', text)
    text = re.sub(r"'", ' <APOSTROPHE> ', text)
    text = re.sub(r'\n', ' <NEWLINE> ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_and_preprocess_data():
    """Load and preprocess data from text8 and MS-MARCO datasets."""
    print("Loading and preprocessing data...")
    
    # Load text8 dataset
    try:
        with open(Config.text8_path, 'r', encoding='utf-8') as f:
            text8_data = f.read()
        text8_data = preprocess_text(text8_data)
        print(f"Loaded text8 dataset: {len(text8_data.split())} words")
    except Exception as e:
        print(f"Error loading text8 dataset: {e}")
        text8_data = ""
    
    # In a real implementation, you would load MS-MARCO here
    # For this example, we'll just use text8
    all_text = text8_data
    
    # Split text into words
    words = all_text.split()
    
    # Count word frequencies
    word_counts = Counter(words)
    
    # Filter words by frequency
    filtered_words = [word for word in words if word_counts[word] >= Config.min_word_freq]
    
    # Create word-to-index and index-to-word mappings
    word_to_idx = {word: idx for idx, (word, _) in enumerate(word_counts.most_common()) 
                   if word_counts[word] >= Config.min_word_freq}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # Save vocabulary
    vocab_df = pd.DataFrame({
        'word': list(word_to_idx.keys()),
        'index': list(word_to_idx.values()),
        'frequency': [word_counts[word] for word in word_to_idx.keys()]
    })
    vocab_df.to_csv(os.path.join(Config.output_dir, 'vocabulary.csv'), index=False)
    
    print(f"Vocabulary size: {len(word_to_idx)}")
    print(f"Total words after filtering: {len(filtered_words)}")
    
    return filtered_words, word_to_idx, idx_to_word

class CBOWDataset(Dataset):
    def __init__(self, words, word_to_idx, window_size, is_test=False):
        self.words = words
        self.word_to_idx = word_to_idx
        self.window_size = window_size
        self.is_test = is_test
        
        # Convert words to indices
        self.word_indices = [self.word_to_idx[word] for word in self.words if word in self.word_to_idx]
        
        # Split data into training and test set (90/10)
        if is_test:
            self.word_indices = self.word_indices[int(0.9 * len(self.word_indices)):]
        else:
            self.word_indices = self.word_indices[:int(0.9 * len(self.word_indices))]
    
    def __len__(self):
        return len(self.word_indices) - 2 * self.window_size
    
    def __getitem__(self, idx):
        # Calculate start and end indices of the context window
        target_idx = idx + self.window_size
        context_indices = []
        
        # Get context indices (words around the target word)
        for i in range(target_idx - self.window_size, target_idx + self.window_size + 1):
            if i != target_idx:  # Skip the target word
                context_indices.append(self.word_indices[i])
        
        # Target word
        target_word_idx = self.word_indices[target_idx]
        
        # Convert to tensors
        context_tensor = torch.tensor(context_indices, dtype=torch.long)
        target_tensor = torch.tensor(target_word_idx, dtype=torch.long)
        
        return context_tensor, target_tensor

class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, use_sparse=True):
        super(CBOWModel, self).__init__()
        
        # Embedding layer (can be sparse or dense)
        if use_sparse:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
        else:
            self.embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=False)
        
        # Linear layer for prediction
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.5 / self.embeddings.embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
    
    def forward(self, contexts):
        # Get embeddings for all context words
        embeds = self.embeddings(contexts)
        
        # Average embeddings (mean pooling)
        hidden = torch.mean(embeds, dim=1)
        
        # Get output scores
        output = self.linear(hidden)
        
        # Apply log softmax for numerical stability
        log_probs = F.log_softmax(output, dim=1)
        
        return log_probs

def find_similar_words(model, word, word_to_idx, idx_to_word, top_k=10):
    """Find most similar words using cosine similarity."""
    if word not in word_to_idx:
        return []
    
    # Get the word index
    word_idx = word_to_idx[word]
    
    # Get the word embedding (needs to be detached from computation graph)
    word_vector = model.embeddings.weight[word_idx].detach().cpu().numpy()
    
    # Calculate cosine similarities for all words
    all_embeddings = model.embeddings.weight.detach().cpu().numpy()
    similarities = np.dot(all_embeddings, word_vector) / (
        np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(word_vector)
    )
    
    # Get indices of most similar words (excluding the word itself)
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    # Return similar words and their similarities
    similar_words = [(idx_to_word[idx], similarities[idx]) for idx in top_indices]
    
    return similar_words

def chunk_vocabulary(vocab_size, num_chunks):
    """Split vocabulary indices into chunks for memory-efficient processing."""
    chunk_size = math.ceil(vocab_size / num_chunks)
    return [range(i * chunk_size, min((i + 1) * chunk_size, vocab_size)) for i in range(num_chunks)]

def train_word2vec():
    """Train the Word2Vec model using CBOW architecture."""
    # Load and preprocess data
    words, word_to_idx, idx_to_word = load_and_preprocess_data()
    
    # Create datasets
    train_dataset = CBOWDataset(words, word_to_idx, Config.window_size, is_test=False)
    test_dataset = CBOWDataset(words, word_to_idx, Config.window_size, is_test=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    vocab_size = len(word_to_idx)
    model = CBOWModel(vocab_size, Config.embedding_dim, use_sparse=Config.use_sparse).to(device)
    
    # Use different optimizers for sparse and dense embeddings
    if Config.use_sparse:
        optimizer = optim.SparseAdam(model.parameters(), lr=Config.learning_rate)
    else:
        optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    
    # Loss function
    criterion = nn.NLLLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.5)
    
    # Configure mixed precision training if enabled
    scaler = torch.cuda.amp.GradScaler(enabled=Config.use_mixed_precision)
    
    # Track best loss
    best_loss = float('inf')
    
    print("Starting training...")
    
    # Training loop
    for epoch in range(Config.epochs):
        print(f"Epoch {epoch+1}/{Config.epochs}")
        model.train()
        
        # Track metrics
        total_loss = 0
        total_batches = 0
        start_time = time.time()
        
        # Initialize gradient accumulation
        optimizer.zero_grad()
        
        # Process batches with progress bar
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (contexts, targets) in progress_bar:
            # Move data to device
            contexts = contexts.to(device)
            targets = targets.to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=Config.use_mixed_precision):
                log_probs = model(contexts)
                loss = criterion(log_probs, targets)
                # Scale loss by gradient accumulation steps
                loss = loss / Config.gradient_accumulation_steps
            
            # Backpropagation with mixed precision
            scaler.scale(loss).backward()
            
            # Accumulate gradients over multiple batches
            if (batch_idx + 1) % Config.gradient_accumulation_steps == 0:
                # Update weights
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Track loss (multiply back by accumulation steps to get true loss)
            total_loss += loss.item() * Config.gradient_accumulation_steps
            total_batches += 1
            
            # Update progress bar
            progress_bar.set_description(f"Loss: {total_loss/total_batches:.4f}")
            
            # Clear cache to save memory
            if batch_idx % 1000 == 0:
                torch.cuda.empty_cache()
            
            # Evaluate and log
            if batch_idx % Config.eval_interval == 0 and batch_idx > 0:
                # Log loss
                wandb.log({
                    "train_loss": total_loss / total_batches,
                    "learning_rate": optimizer.param_groups[0]['lr'],
                    "batch": batch_idx + epoch * len(train_loader)
                })
                
                # Find similar words for test words
                model.eval()
                for test_word in Config.test_words:
                    if test_word in word_to_idx:
                        similar_words = find_similar_words(model, test_word, word_to_idx, idx_to_word, Config.top_k)
                        print(f"\nSimilar words to '{test_word}':")
                        for word, similarity in similar_words:
                            print(f"  {word}: {similarity:.4f}")
                model.train()
        
        # Calculate average loss for epoch
        epoch_loss = total_loss / total_batches
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Loss: {epoch_loss:.4f}")
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        test_batches = 0
        
        with torch.no_grad():
            for contexts, targets in tqdm(test_loader, desc="Testing"):
                contexts = contexts.to(device)
                targets = targets.to(device)
                
                log_probs = model(contexts)
                loss = criterion(log_probs, targets)
                
                test_loss += loss.item()
                test_batches += 1
        
        avg_test_loss = test_loss / test_batches
        print(f"Test Loss: {avg_test_loss:.4f}")
        
        # Update learning rate
        scheduler.step(avg_test_loss)
        
        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "test_loss": avg_test_loss,
            "epoch_time": epoch_time
        })
        
        # Save checkpoint if it's the best model
        if avg_test_loss < best_loss:
            best_loss = avg_test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'word_to_idx': word_to_idx,
                'idx_to_word': idx_to_word
            }, os.path.join(Config.output_dir, 'best_model.pt'))
            print("Saved new best model")
    
    print("Training complete!")
    
    # Save final embeddings
    print("Saving embeddings...")
    embeddings = model.embeddings.weight.data.cpu().numpy()
    np.save(os.path.join(Config.output_dir, 'word2vec_embeddings.npy'), embeddings)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(Config.output_dir, 'final_model.pt'))
    
    # Create word vectors file in text format (word and its embedding)
    with open(os.path.join(Config.output_dir, 'word_vectors.txt'), 'w', encoding='utf-8') as f:
        f.write(f"{len(word_to_idx)} {Config.embedding_dim}\n")
        for word, idx in word_to_idx.items():
            vector_str = ' '.join(str(val) for val in embeddings[idx])
            f.write(f"{word} {vector_str}\n")
    
    print(f"All files saved to {Config.output_dir}")
    
    return model, word_to_idx, idx_to_word

if __name__ == "__main__":
    model, word_to_idx, idx_to_word = train_word2vec()
    
    # Evaluate final model on test words
    print("\n=== Final Model Evaluation ===")
    for test_word in Config.test_words:
        if test_word in word_to_idx:
            similar_words = find_similar_words(model, test_word, word_to_idx, idx_to_word, Config.top_k)
            print(f"\nSimilar words to '{test_word}':")
            for word, similarity in similar_words:
                print(f"  {word}: {similarity:.4f}")
    
    # Complete W&B run
    wandb.finish()