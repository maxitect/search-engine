"""
Main script for training Word2Vec model on text8 dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
import gc

from src.config import Config
from src.models.word2vec import Word2Vec
from src.utils.data import load_text8_data, build_vocabulary, create_data_loader

def print_memory_usage():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def train_model(model: Word2Vec, train_loader: torch.utils.data.DataLoader, 
                device: torch.device, config: Config) -> Word2Vec:
    """
    Train the Word2Vec model.
    
    Args:
        model (Word2Vec): Word2Vec model
        train_loader (DataLoader): Training data loader
        device (torch.device): Device to train on
        config (Config): Configuration object
    
    Returns:
        Word2Vec: Trained model
    """
    # Set up training
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.NLLLoss()
    
    # Training loop
    for epoch in range(config.epochs):
        print(f"\nEpoch {epoch + 1}/{config.epochs}")
        total_loss = 0
        optimizer.zero_grad()  # Zero gradients at the start of epoch
        
        for i, (contexts, targets) in enumerate(tqdm(train_loader)):
            # Move to device
            contexts = contexts.to(device)
            targets = targets.to(device)
            
            # Forward pass
            output = model(contexts)
            loss = criterion(output, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / config.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Step optimizer only after accumulating gradients
            if (i + 1) % config.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                
                # Clear memory
                torch.cuda.empty_cache()
                gc.collect()
            
            total_loss += loss.item() * config.gradient_accumulation_steps
            
            # Print memory usage every 1000 batches
            if (i + 1) % 1000 == 0:
                print_memory_usage()
        
        # Handle remaining gradients
        if len(train_loader) % config.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Average loss: {avg_loss:.4f}")
        
        # Clear memory at the end of each epoch
        torch.cuda.empty_cache()
        gc.collect()
    
    return model

def main():
    # Load configuration
    config = Config()
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if text8 file exists
    if not os.path.exists(config.data_path):
        print(f"Error: text8 file not found at {config.data_path}")
        print("Please make sure the text8 file is in the data directory")
        return
    
    # Load and preprocess data
    print("Loading text8 data...")
    words = load_text8_data(config.data_path)
    print(f"Total words: {len(words)}")
    
    # Build vocabulary
    vocab, word_to_idx, idx_to_word = build_vocabulary(
        words, 
        min_freq=config.min_word_freq,
        max_vocab_size=config.max_vocab_size
    )
    
    # Filter words to only include vocabulary words
    filtered_words = [word for word in words if word in word_to_idx]
    print(f"Words after filtering: {len(filtered_words)}")
    
    # Clear memory before creating model
    torch.cuda.empty_cache()
    gc.collect()
    print_memory_usage()
    
    # Create model
    print("Creating model...")
    model = Word2Vec(len(vocab), config.embedding_dim)
    model = model.to(device)
    print_memory_usage()
    
    # Create data loader
    print("Creating data loader...")
    train_loader = create_data_loader(filtered_words, word_to_idx, config.window_size, config.batch_size)
    print_memory_usage()
    
    # Train model
    print("Starting training...")
    model = train_model(model, train_loader, device, config)
    
    # Test the model
    test_words = ["computer", "technology", "data", "learning", "system"]
    print("\nTesting similar words:")
    for word in test_words:
        if word in word_to_idx:
            word_idx = word_to_idx[word]
            similar_words = model.find_similar_words(word_idx, word_to_idx, idx_to_word)
            print(f"\nSimilar words to '{word}':")
            for similar_word, similarity in similar_words:
                print(f"  {similar_word}: {similarity:.4f}")
    
    # Save model and embeddings
    print("\nSaving model and embeddings...")
    torch.save(model.state_dict(), os.path.join(config.output_dir, "model.pt"))
    
    # Save embeddings
    embeddings = model.embeddings.weight.data.cpu().numpy()
    np.save(os.path.join(config.output_dir, "embeddings.npy"), embeddings)
    
    # Save vocabulary
    with open(os.path.join(config.output_dir, "vocabulary.txt"), "w", encoding="utf-8") as f:
        for word, idx in word_to_idx.items():
            f.write(f"{word}\t{idx}\n")
    
    print("Done!")

if __name__ == "__main__":
    main() 