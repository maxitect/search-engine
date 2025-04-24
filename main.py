"""
Simple Word2Vec implementation for text8 and MS-MARCO datasets with memory optimization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np
from tqdm import tqdm
import os
import json
import gc

# Configuration
WINDOW_SIZE = 5
MIN_WORD_FREQ = 5
EMBEDDING_DIM = 8  # Reduced for memory efficiency
BATCH_SIZE = 64  # Reduced batch size
LEARNING_RATE = 0.001
EPOCHS = 5
CHUNK_SIZE = 1000000  # Process 1M words at a time
GRADIENT_ACCUMULATION_STEPS = 4  # Accumulate gradients

class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
        # Initialize weights
        self.embeddings.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.zero_()
    
    def forward(self, contexts):
        # Get embeddings for context words
        embeds = self.embeddings(contexts)
        # Average embeddings
        embeds = embeds.mean(dim=1)
        # Get output scores
        output = self.linear(embeds)
        return torch.log_softmax(output, dim=1)

def load_text8_data(file_path):
    """Load and preprocess text8 data in chunks."""
    print("Loading text8 data...")
    words = []
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            words.extend(chunk.lower().split())
            print(f"Processed {len(words)} words...")
    return words

def load_msmarco_data(file_path):
    """Load and preprocess MS-MARCO data in chunks."""
    print("Loading MS-MARCO data...")
    words = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            data = json.loads(line)
            # Process query
            words.extend(data['query'].lower().split())
            # Process passages
            for passage in data['passages']:
                words.extend(passage['passage_text'].lower().split())
            
            # Clear memory periodically
            if len(words) % CHUNK_SIZE == 0:
                gc.collect()
                torch.cuda.empty_cache()
    return words

def build_vocabulary(words):
    """Build vocabulary from words."""
    print("Building vocabulary...")
    word_counts = Counter(words)
    vocab = [word for word, count in word_counts.items() if count >= MIN_WORD_FREQ]
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    return vocab, word_to_idx, idx_to_word

def create_context_target_pairs(words, word_to_idx, start_idx, end_idx):
    """Create context-target pairs for a chunk of words."""
    pairs = []
    for i in range(start_idx + WINDOW_SIZE, end_idx - WINDOW_SIZE):
        target = word_to_idx[words[i]]
        context = [word_to_idx[words[j]] for j in range(i - WINDOW_SIZE, i + WINDOW_SIZE + 1) 
                  if j != i]
        pairs.append((context, target))
    return pairs

def train_on_chunk(model, words, word_to_idx, start_idx, end_idx, device):
    """Train model on a chunk of data."""
    # Create training pairs for this chunk
    train_data = create_context_target_pairs(words, word_to_idx, start_idx, end_idx)
    
    # Convert to tensors
    contexts = torch.tensor([pair[0] for pair in train_data], dtype=torch.long)
    targets = torch.tensor([pair[1] for pair in train_data], dtype=torch.long)
    
    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(contexts, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Training setup
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()
    
    # Training loop
    total_loss = 0
    optimizer.zero_grad()
    
    for batch_idx, (contexts, targets) in enumerate(tqdm(dataloader)):
        # Move to device
        contexts = contexts.to(device)
        targets = targets.to(device)
        
        # Forward pass
        output = model(contexts)
        loss = criterion(output, targets)
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass
        loss.backward()
        
        # Accumulate gradients
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # Clear memory
            gc.collect()
            torch.cuda.empty_cache()
    
    # Handle remaining gradients
    if len(dataloader) % GRADIENT_ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return total_loss / len(dataloader)

def train_model(model, words, word_to_idx, device):
    """Train the Word2Vec model in chunks."""
    print("Starting training...")
    num_chunks = (len(words) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        total_loss = 0
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * CHUNK_SIZE
            end_idx = min((chunk_idx + 1) * CHUNK_SIZE, len(words))
            
            print(f"Processing chunk {chunk_idx + 1}/{num_chunks}")
            chunk_loss = train_on_chunk(model, words, word_to_idx, start_idx, end_idx, device)
            total_loss += chunk_loss
            
            # Clear memory between chunks
            gc.collect()
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_chunks
        print(f"Average loss: {avg_loss:.4f}")
    
    return model

def find_similar_words(model, word, word_to_idx, idx_to_word, top_k=10):
    """Find most similar words using cosine similarity."""
    if word not in word_to_idx:
        return []
    
    # Get the word embedding
    word_idx = word_to_idx[word]
    word_vector = model.embeddings.weight[word_idx].detach().cpu().numpy()
    
    # Calculate similarities
    similarities = []
    for idx, emb in enumerate(model.embeddings.weight):
        if idx != word_idx:
            sim = np.dot(word_vector, emb.detach().cpu().numpy())
            similarities.append((idx_to_word[idx], sim))
    
    # Sort by similarity and get top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def main():
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and preprocess data
    text8_path = "data/text8"
    msmarco_path = "data/msmarco/train_v2.1.json"
    
    if not os.path.exists(text8_path):
        print(f"Error: text8 file not found at {text8_path}")
        return
    
    if not os.path.exists(msmarco_path):
        print(f"Error: MS-MARCO file not found at {msmarco_path}")
        return
    
    # Load data
    text8_words = load_text8_data(text8_path)
    msmarco_words = load_msmarco_data(msmarco_path)
    all_words = text8_words + msmarco_words
    
    # Build vocabulary
    vocab, word_to_idx, idx_to_word = build_vocabulary(all_words)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Create model
    model = Word2Vec(len(vocab), EMBEDDING_DIM).to(device)
    
    # Train model
    model = train_model(model, all_words, word_to_idx, device)
    
    # Test the model
    test_words = ["computer", "technology", "data", "learning", "system", "search", "query", "document"]
    print("\nTesting similar words:")
    for word in test_words:
        if word in word_to_idx:
            similar_words = find_similar_words(model, word, word_to_idx, idx_to_word)
            print(f"\nSimilar words to '{word}':")
            for similar_word, similarity in similar_words:
                print(f"  {similar_word}: {similarity:.4f}")
    
    # Save model and embeddings
    print("\nSaving model and embeddings...")
    torch.save(model.state_dict(), "model.pt")
    embeddings = model.embeddings.weight.data.cpu().numpy()
    np.save("embeddings.npy", embeddings)
    
    # Save vocabulary
    with open("vocabulary.txt", "w", encoding="utf-8") as f:
        for word, idx in word_to_idx.items():
            f.write(f"{word}\t{idx}\n")
    
    print("Done!")

if __name__ == "__main__":
    main() 