"""
Training module for Word2Vec model.
"""

import os
import gc
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb
from typing import Dict, List, Tuple
import numpy as np

from ..config import Config
from ..models.word2vec import Word2VecModel
from ..utils.memory import calculate_dynamic_batch_size, clear_memory

class SubsampledMemoryEfficientDataset(Dataset):
    """Memory-efficient dataset that generates context-target pairs on-the-fly."""
    
    def __init__(self, words: List[str], word_to_idx: Dict[str, int], 
                 window_size: int, is_test: bool = False):
        self.words = words
        self.word_to_idx = word_to_idx
        self.window_size = window_size
        
        # Convert words to indices
        self.word_indices = [word_to_idx[word] for word in words if word in word_to_idx]
        
        # Split data into training and test set
        split_idx = int(0.9 * len(self.word_indices))
        if is_test:
            self.word_indices = self.word_indices[split_idx:]
        else:
            self.word_indices = self.word_indices[:split_idx]
    
    def __len__(self) -> int:
        return max(0, len(self.word_indices) - 2 * self.window_size)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        target_idx = idx + self.window_size
        context_indices = []
        
        # Get context indices
        for i in range(target_idx - self.window_size, target_idx + self.window_size + 1):
            if i != target_idx and 0 <= i < len(self.word_indices):
                context_indices.append(self.word_indices[i])
        
        # Target word
        target_word_idx = self.word_indices[target_idx]
        
        # Convert to tensors
        context_tensor = torch.tensor(context_indices, dtype=torch.long)
        target_tensor = torch.tensor(target_word_idx, dtype=torch.long)
        
        return context_tensor, target_tensor

def train_on_chunk(model: Word2VecModel, data_chunk: List[str], 
                  optimizer: optim.Optimizer, criterion: nn.Module,
                  scaler: torch.cuda.amp.GradScaler, device: torch.device,
                  gradient_accumulation_steps: int, use_mixed_precision: bool) -> float:
    """Train model on a chunk of data with memory optimization."""
    # Calculate dynamic batch size
    batch_size = calculate_dynamic_batch_size(model, device, Config.memory_safety_factor)
    print(f"Using dynamic batch size: {batch_size}")
    
    # Create dataset and loader
    chunk_dataset = SubsampledMemoryEfficientDataset(data_chunk, word_to_idx, Config.window_size)
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
            clear_memory()
        
        # Track loss
        total_loss += loss.item() * gradient_accumulation_steps
        total_batches += 1
        
        # Update progress bar
        progress_bar.set_description(f"Loss: {total_loss/total_batches:.4f}")
        
        # Clear memory periodically
        if batch_idx % 5 == 0:  # More frequent memory cleanup
            clear_memory()
    
    # Make sure to update any remaining accumulated gradients
    if total_batches % gradient_accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    
    return total_loss / max(1, total_batches)

def train_word2vec(filtered_words: List[str], word_to_idx: Dict[str, int],
                  idx_to_word: Dict[int, str], device: torch.device) -> Tuple[Word2VecModel, Dict[str, int], Dict[int, str]]:
    """Train the Word2Vec model with memory optimization."""
    # Create model
    vocab_size = len(word_to_idx)
    print(f"Creating model with vocabulary size: {vocab_size}")
    
    # Free memory before creating model
    clear_memory()
    
    # Initialize model
    model = Word2VecModel(
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
                        similar_words = model.find_similar_words(test_word, word_to_idx, idx_to_word, Config.top_k)
                        print(f"\nSimilar words to '{test_word}':")
                        for word, similarity in similar_words:
                            print(f"  {word}: {similarity:.4f}")
                model.train()
            
            # Clear memory between chunks
            clear_memory()
        
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
    chunk_size = 1000  # Smaller chunk size for saving
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