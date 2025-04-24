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
wandb.init(project="word2vec-training", name="cbow-memory-optimized")

class Config:
    # Data parameters
    window_size = 5
    min_word_freq = 30  # Increased to reduce vocabulary
    max_vocab_size = 10000  # Reduced from 25000
    
    # Model parameters
    embedding_dim = 16  # Reduced from 25
    learning_rate = 0.001
    initial_batch_size = 128  # Reduced from 256
    epochs = 5
    
    # Training optimizations
    use_sparse = True
    use_mixed_precision = True
    gradient_accumulation_steps = 128  # Increased from 64
    chunk_size = 100000  # Reduced from 250000
    memory_safety_factor = 0.5  # More conservative safety factor
    
    # Paths
    text8_path = "data/text8"
    output_dir = "data/word2vec"
    
    # Evaluation
    test_words = ["computer", "technology", "data", "learning", "system"]
    eval_interval = 1000
    top_k = 10

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
    """Load and preprocess data from text8 dataset with memory optimization."""
    print("Loading and preprocessing data...")
    
    # Load text8 dataset in chunks to save memory
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
                
                # Print progress
                print(f"Processed {len(words)} words...")
                
                # Force garbage collection
                gc.collect()
        
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
    
    # Save vocabulary
    vocab_df = pd.DataFrame({
        'word': list(word_to_idx.keys()),
        'index': list(word_to_idx.values()),
        'frequency': [word_counts[word] for word in word_to_idx.keys()]
    })
    vocab_df.to_csv(os.path.join(Config.output_dir, 'vocabulary.csv'), index=False)
    
    # Filter words to only include vocabulary words to save memory
    filtered_words = [word for word in words if word in word_to_idx]
    
    print(f"Vocabulary size: {len(word_to_idx)}")
    print(f"Total words after filtering: {len(filtered_words)}")
    
    # Clear memory
    del words
    gc.collect()
    
    return filtered_words, word_to_idx, idx_to_word

class SubsampledMemoryEfficientDataset(Dataset):
    """Memory-efficient dataset that generates context-target pairs on-the-fly."""
    def __init__(self, words, word_to_idx, window_size, is_test=False):
        self.words = words
        self.word_to_idx = word_to_idx
        self.window_size = window_size
        
        # Convert words to indices (only store indices to save memory)
        print("Converting words to indices...")
        self.word_indices = np.array([word_to_idx[word] for word in words if word in word_to_idx], dtype=np.int32)
        del words  # Free memory
        
        # Subsample frequent words to improve quality and reduce dataset size
        print("Subsampling frequent words...")
        self.word_indices = self._subsample_frequent_words(self.word_indices)
        
        # Split data into training and test set (90/10)
        split_idx = int(0.9 * len(self.word_indices))
        if is_test:
            self.word_indices = self.word_indices[split_idx:]
        else:
            self.word_indices = self.word_indices[:split_idx]
        
        print(f"Dataset size: {len(self.word_indices)}")
    
    def _subsample_frequent_words(self, word_indices):
        """Subsample frequent words according to word2vec paper."""
        # Count frequencies
        word_counts = Counter(word_indices)
        total_words = len(word_indices)
        
        # Calculate subsampling probabilities
        threshold = 1e-5
        word_freq = {word: count / total_words for word, count in word_counts.items()}
        keep_prob = {word: (np.sqrt(word_freq[word] / threshold) + 1) * (threshold / word_freq[word])
                    for word in word_freq if word_freq[word] > threshold}
        
        # Apply subsampling (keep words based on probability)
        keep_indices = np.random.random(len(word_indices)) < np.array(
            [keep_prob.get(word, 1.0) for word in word_indices])
        
        subsampled_indices = word_indices[keep_indices]
        print(f"Subsampling reduced data from {len(word_indices)} to {len(subsampled_indices)} words")
        
        return subsampled_indices
    
    def __len__(self):
        # Account for window size at both ends
        return max(0, len(self.word_indices) - 2 * self.window_size)
    
    def __getitem__(self, idx):
        # Calculate start and end indices of the context window
        target_idx = idx + self.window_size
        context_indices = []
        
        # Get context indices (words around the target word)
        for i in range(target_idx - self.window_size, target_idx + self.window_size + 1):
            if i != target_idx and 0 <= i < len(self.word_indices):  # Skip target word & check bounds
                context_indices.append(self.word_indices[i])
        
        # Target word
        target_word_idx = self.word_indices[target_idx]
        
        # Convert to tensors
        context_tensor = torch.tensor(context_indices, dtype=torch.long)
        target_tensor = torch.tensor(target_word_idx, dtype=torch.long)
        
        return context_tensor, target_tensor

def get_gpu_memory():
    """Get available GPU memory in MB"""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
    return 0

def calculate_dynamic_batch_size(model, device, safety_factor=0.5):
    """Calculate optimal batch size based on available memory"""
    total_memory = get_gpu_memory()
    if total_memory == 0:
        return Config.initial_batch_size
    
    # Estimate memory per sample (more conservative estimate)
    sample_size = (Config.embedding_dim * Config.window_size * 4)  # 4 bytes per float
    # Add overhead for model parameters and gradients
    overhead = (Config.embedding_dim * Config.max_vocab_size * 4) / 1024  # KB
    available_memory = (total_memory * safety_factor) - overhead
    
    max_batch_size = int(available_memory / sample_size)
    
    # Ensure batch size is a power of 2 and within reasonable limits
    batch_size = min(2 ** int(np.log2(max_batch_size)), Config.initial_batch_size)
    return max(16, batch_size)  # Minimum batch size of 16

class MemoryEfficientCBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, use_sparse=True):
        super(MemoryEfficientCBOWModel, self).__init__()
        
        # Split vocabulary into smaller chunks
        self.chunk_size = 2000  # Reduced from 5000
        self.num_chunks = (vocab_size + self.chunk_size - 1) // self.chunk_size
        
        # Create embedding chunks
        self.embeddings = nn.ModuleList([
            nn.Embedding(min(self.chunk_size, vocab_size - i * self.chunk_size), 
                        embedding_dim, sparse=use_sparse)
            for i in range(self.num_chunks)
        ])
        
        # Linear layer with tied weights
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=False)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        for emb in self.embeddings:
            emb.weight.data.uniform_(-initrange, initrange)
        self.linear.weight.data.uniform_(-initrange, initrange)
    
    def get_embedding(self, indices):
        """Get embeddings from appropriate chunk with memory optimization"""
        embeddings = []
        for i in range(self.num_chunks):
            mask = (indices >= i * self.chunk_size) & (indices < (i + 1) * self.chunk_size)
            if mask.any():
                chunk_indices = indices[mask] - i * self.chunk_size
                chunk_emb = self.embeddings[i](chunk_indices)
                embeddings.append((mask, chunk_emb))
        
        if not embeddings:
            return torch.zeros(len(indices), self.embeddings[0].embedding_dim, device=indices.device)
        
        result = torch.zeros(len(indices), self.embeddings[0].embedding_dim, device=indices.device)
        for mask, emb in embeddings:
            result[mask] = emb
        return result
    
    def forward(self, contexts):
        if self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, contexts)
        return self._forward(contexts)
    
    def _forward(self, contexts):
        # Get embeddings for context
        embeds = self.get_embedding(contexts)
        embeds = embeds.view(-1, contexts.size(1), self.embeddings[0].embedding_dim)
        embeds = embeds.mean(dim=1)
        
        # Get output scores
        output = self.linear(embeds)
        return F.log_softmax(output, dim=1)

def train_on_chunk(model, data_chunk, optimizer, criterion, scaler, device, 
                  gradient_accumulation_steps, use_mixed_precision):
    """Train model on a chunk of data with memory optimization"""
    # Calculate dynamic batch size
    batch_size = calculate_dynamic_batch_size(model, device, Config.memory_safety_factor)
    print(f"Using dynamic batch size: {batch_size}")
    
    # Create dataset and loader for this chunk
    chunk_dataset = SubsampledMemoryEfficientDataset(data_chunk, word_to_idx, Config.window_size, is_test=False)
    chunk_loader = DataLoader(chunk_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Track metrics
    total_loss = 0
    total_batches = 0
    
    # Training loop
    model.train()
    progress_bar = tqdm(enumerate(chunk_loader), total=len(chunk_loader), desc="Training on chunk")
    
    for batch_idx, (contexts, targets) in progress_bar:
        # Move data to device
        contexts = contexts.to(device)
        targets = targets.to(device)
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            log_probs = model(contexts)
            loss = criterion(log_probs, targets)
            loss = loss / gradient_accumulation_steps
        
        # Backpropagation with mixed precision
        scaler.scale(loss).backward()
        
        # Accumulate gradients
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            # Clear cache after each gradient step
            torch.cuda.empty_cache()
            gc.collect()
        
        # Track loss
        total_loss += loss.item() * gradient_accumulation_steps
        total_batches += 1
        
        # Update progress bar
        progress_bar.set_description(f"Loss: {total_loss/total_batches:.4f}")
        
        # Clear memory periodically
        if batch_idx % 10 == 0:  # More frequent cleanup
            torch.cuda.empty_cache()
            gc.collect()
    
    # Make sure to update any remaining accumulated gradients
    if total_batches % gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return total_loss / max(1, total_batches)

def find_similar_words(model, word, word_to_idx, idx_to_word, top_k=10):
    """Find most similar words using cosine similarity."""
    if word not in word_to_idx:
        return []
    
    # Get the word index
    word_idx = word_to_idx[word]
    
    # Get the word embedding
    word_vector = model.embeddings.weight[word_idx].detach().cpu().numpy()
    
    # Calculate similarities for a subset of words to save memory
    embeddings = model.embeddings.weight.detach().cpu().numpy()
    
    # Calculate cosine similarities
    similarities = np.dot(embeddings, word_vector) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(word_vector)
    )
    
    # Get top k indices
    top_indices = np.argsort(similarities)[::-1][1:top_k+1]
    
    # Return similar words
    similar_words = [(idx_to_word[idx], similarities[idx]) for idx in top_indices]
    
    return similar_words

def train_word2vec():
    """Train the Word2Vec model with memory optimization."""
    # Load and preprocess data
    filtered_words, word_to_idx, idx_to_word = load_and_preprocess_data()
    
    # Create model
    vocab_size = len(word_to_idx)
    print(f"Creating model with vocabulary size: {vocab_size}")
    
    # Free memory before creating model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Check available memory
    total_memory = get_gpu_memory()
    print(f"Total GPU memory: {total_memory:.2f} MB")
    
    # Initialize model
    model = MemoryEfficientCBOWModel(
        vocab_size=vocab_size, 
        embedding_dim=Config.embedding_dim, 
        use_sparse=Config.use_sparse
    ).to(device)
    
    # Configure optimizer
    optimizer = optim.SparseAdam(model.parameters(), lr=Config.learning_rate)
    
    # Loss function
    criterion = nn.NLLLoss()
    
    # Configure mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=Config.use_mixed_precision)
    
    # Track best loss
    best_loss = float('inf')
    
    # Split data into chunks
    num_chunks = (len(filtered_words) + Config.chunk_size - 1) // Config.chunk_size
    print(f"Training on {num_chunks} chunks of data...")
    
    # Training loop
    for epoch in range(Config.epochs):
        print(f"Epoch {epoch+1}/{Config.epochs}")
        epoch_start_time = time.time()
        epoch_loss = 0
        
        # Process each chunk
        for chunk_idx in range(num_chunks):
            print(f"Processing chunk {chunk_idx+1}/{num_chunks}")
            chunk_start = chunk_idx * Config.chunk_size
            chunk_end = min((chunk_idx + 1) * Config.chunk_size, len(filtered_words))
            data_chunk = filtered_words[chunk_start:chunk_end]
            
            # Train on this chunk
            chunk_loss = train_on_chunk(
                model=model,
                data_chunk=data_chunk,
                optimizer=optimizer,
                criterion=criterion,
                scaler=scaler,
                device=device,
                gradient_accumulation_steps=Config.gradient_accumulation_steps,
                use_mixed_precision=Config.use_mixed_precision
            )
            
            epoch_loss += chunk_loss * (chunk_end - chunk_start)
            
            # Log to wandb
            wandb.log({
                "chunk_loss": chunk_loss,
                "chunk": chunk_idx + epoch * num_chunks
            })
            
            # Evaluate every few chunks
            if chunk_idx % 2 == 0:
                model.eval()
                for test_word in Config.test_words:
                    if test_word in word_to_idx:
                        similar_words = find_similar_words(model, test_word, word_to_idx, idx_to_word, Config.top_k)
                        print(f"\nSimilar words to '{test_word}':")
                        for word, similarity in similar_words:
                            print(f"  {word}: {similarity:.4f}")
                model.train()
            
            # Clear memory between chunks
            gc.collect()
            torch.cuda.empty_cache()
        
        # Calculate average loss for epoch
        avg_epoch_loss = epoch_loss / len(filtered_words)
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s, Loss: {avg_epoch_loss:.4f}")
        
        # Save checkpoint if it's the best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
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
    
    # Save in chunks
    chunk_size = 2000  # Reduced from 5000
    for i in range(0, len(word_to_idx), chunk_size):
        chunk_end = min(i + chunk_size, len(word_to_idx))
        chunk_embeddings = embeddings[i:chunk_end]
        np.save(os.path.join(Config.output_dir, f'word2vec_embeddings_chunk_{i}.npy'), chunk_embeddings)
    
    # Save vocabulary
    with open(os.path.join(Config.output_dir, 'vocabulary.txt'), 'w', encoding='utf-8') as f:
        for word, idx in word_to_idx.items():
            f.write(f"{word}\t{idx}\n")
    
    # Save model
    torch.save(model.state_dict(), os.path.join(Config.output_dir, 'final_model.pt'))
    
    print(f"All files saved to {Config.output_dir}")
    
    return model, word_to_idx, idx_to_word

if __name__ == "__main__":
    try:
        # Try to free as much memory as possible
        gc.collect()
        torch.cuda.empty_cache()
        
        model, word_to_idx, idx_to_word = train_word2vec()
        
        # Evaluate final model on test words
        print("\n=== Final Model Evaluation ===")
        for test_word in Config.test_words:
            if test_word in word_to_idx:
                similar_words = find_similar_words(model, test_word, word_to_idx, idx_to_word, Config.top_k)
                print(f"\nSimilar words to '{test_word}':")
                for word, similarity in similar_words:
                    print(f"  {word}: {similarity:.4f}")
        
    except Exception as e:
        print(f"Error during training: {e}")
    finally:
        # Complete W&B run
        wandb.finish()