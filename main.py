"""
Simple Word2Vec implementation for text8 and MS-MARCO datasets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import numpy as np
from tqdm import tqdm
import os
import json

# Configuration
WINDOW_SIZE = 5
MIN_WORD_FREQ = 5  # Reduced to include more words
EMBEDDING_DIM = 16  # Increased for better word representation
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 5

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
    """Load and preprocess text8 data."""
    print("Loading text8 data...")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().lower()
    return text.split()

def load_msmarco_data(file_path):
    """Load and preprocess MS-MARCO data."""
    print("Loading MS-MARCO data...")
    words = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Process query
            words.extend(data['query'].lower().split())
            # Process passages
            for passage in data['passages']:
                words.extend(passage['passage_text'].lower().split())
    return words

def load_data(text8_path, msmarco_path):
    """Load and preprocess both datasets."""
    # Load text8 data
    text8_words = load_text8_data(text8_path)
    print(f"Text8 words: {len(text8_words)}")
    
    # Load MS-MARCO data
    msmarco_words = load_msmarco_data(msmarco_path)
    print(f"MS-MARCO words: {len(msmarco_words)}")
    
    # Combine datasets
    all_words = text8_words + msmarco_words
    print(f"Total words: {len(all_words)}")
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Filter vocabulary by frequency only
    vocab = [word for word, count in word_counts.items() if count >= MIN_WORD_FREQ]
    
    # Create word to index mapping
    word_to_idx = {word: idx for idx, word in enumerate(vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # Filter words to only include vocabulary words
    filtered_words = [word for word in all_words if word in word_to_idx]
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Words after filtering: {len(filtered_words)}")
    
    return filtered_words, word_to_idx, idx_to_word

def create_context_target_pairs(words, word_to_idx):
    """Create context-target pairs for training."""
    pairs = []
    for i in range(WINDOW_SIZE, len(words) - WINDOW_SIZE):
        target = word_to_idx[words[i]]
        context = [word_to_idx[words[j]] for j in range(i - WINDOW_SIZE, i + WINDOW_SIZE + 1) 
                  if j != i]
        pairs.append((context, target))
    return pairs

def train_model(model, train_data, device):
    """Train the Word2Vec model."""
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.NLLLoss()
    
    # Convert to tensors
    contexts = torch.tensor([pair[0] for pair in train_data], dtype=torch.long)
    targets = torch.tensor([pair[1] for pair in train_data], dtype=torch.long)
    
    # Create DataLoader
    dataset = torch.utils.data.TensorDataset(contexts, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Training loop
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        total_loss = 0
        
        for contexts, targets in tqdm(dataloader):
            # Move to device
            contexts = contexts.to(device)
            targets = targets.to(device)
            
            # Forward pass
            output = model(contexts)
            loss = criterion(output, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
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
    
    words, word_to_idx, idx_to_word = load_data(text8_path, msmarco_path)
    
    # Create training data
    print("Creating training pairs...")
    train_data = create_context_target_pairs(words, word_to_idx)
    
    # Create model
    model = Word2Vec(len(word_to_idx), EMBEDDING_DIM).to(device)
    
    # Train model
    print("Training model...")
    model = train_model(model, train_data, device)
    
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