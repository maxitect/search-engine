import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from datasets import load_dataset
from collections import Counter
import numpy as np
from tqdm import tqdm
import wandb
import os
import re
import time

def clean_text(text):
    """Basic text cleaning"""
    # Convert to lowercase
    text = text.lower()
    
    # Replace punctuation with spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Split into words and remove empty strings
    words = [word for word in text.split() if word]
    
    return words

def load_text8():
    """Load and clean text8 dataset"""
    print("Loading text8...")
    with open("data/text8", "r") as f:
        text = f.read()
    words = clean_text(text)
    print(f"Loaded {len(words)} words from text8")
    return words

def load_msmarco():
    """Load and clean MS-MARCO dataset"""
    print("Loading MS-MARCO...")
    dataset = load_dataset("ms_marco", "v1.1", split="train")
    words = []
    
    # Process in chunks to manage memory
    chunk_size = 10000
    total_examples = len(dataset)
    print(f"Total MS-MARCO examples: {total_examples}")
    
    for chunk_start in tqdm(range(0, total_examples, chunk_size), desc="Processing MS-MARCO"):
        chunk_end = min(chunk_start + chunk_size, total_examples)
        chunk = dataset.select(range(chunk_start, chunk_end))
        
        chunk_words = []
        for example in chunk:
            passages = example['passages']['passage_text']
            for passage in passages:
                chunk_words.extend(clean_text(passage))
        
        words.extend(chunk_words)
        print(f"Processed {chunk_end}/{total_examples} examples, current word count: {len(words)}")
    
    print(f"Loaded {len(words)} words from MS-MARCO")
    return words

def combine_datasets():
    """Combine text8 and MS-MARCO datasets"""
    text8_words = load_text8()
    msmarco_words = load_msmarco()
    
    # Combine words
    all_words = text8_words + msmarco_words
    
    # Build vocabulary
    word_counts = Counter(all_words)
    vocab = [word for word, count in word_counts.items() if count >= 5]  # Basic filtering
    
    print(f"Total words after combining: {len(all_words)}")
    print(f"Vocabulary size: {len(vocab)}")
    
    return all_words, vocab

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
        
        # Initialize weights
        self.embeddings.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        self.linear.weight.data.uniform_(-0.5 / embedding_dim, 0.5 / embedding_dim)
        self.linear.bias.data.zero_()
        
    def forward(self, context):
        # context shape: (batch_size, window_size*2)
        embeds = self.embeddings(context)  # (batch_size, window_size*2, embedding_dim)
        embeds = embeds.mean(dim=1)        # (batch_size, embedding_dim)
        return self.linear(embeds)         # (batch_size, vocab_size)

def get_similar_words(model, word, word2idx, idx2word, top_k=5):
    """Get similar words using cosine similarity"""
    if word not in word2idx:
        return f"Word '{word}' not in vocabulary"
    
    word_idx = word2idx[word]
    word_embedding = model.embeddings.weight[word_idx]
    
    # Calculate cosine similarity with all words
    similarities = F.cosine_similarity(
        word_embedding.unsqueeze(0),
        model.embeddings.weight,
        dim=1
    )
    
    # Get top k similar words (excluding the word itself)
    top_k_indices = torch.topk(similarities, k=top_k+1)[1][1:]  # +1 to exclude the word itself
    similar_words = [(idx2word[idx.item()], similarities[idx].item()) for idx in top_k_indices]
    
    return similar_words

def train():
    # Initialize wandb
    wandb.init(project="word2vec-cbow", config={
        "architecture": "CBOW",
        "dataset": "text8+MS-MARCO",
        "embedding_dim": 300,
        "window_size": 5,
        "batch_size": 4096,
        "learning_rate": 0.001,
        "epochs": 10
    })
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and combine datasets
    print("Loading and combining datasets...")
    words, vocab = combine_datasets()
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    # Create training pairs
    print("Creating training pairs...")
    window_size = wandb.config.window_size
    train_data = []
    
    # Process in chunks to manage memory
    chunk_size = 1000000
    for i in tqdm(range(0, len(words), chunk_size), desc="Creating training pairs"):
        chunk_end = min(i + chunk_size, len(words))
        chunk = words[i:chunk_end]
        
        for j in range(window_size, len(chunk)-window_size):
            context = chunk[j-window_size:j] + chunk[j+1:j+window_size+1]
            target = chunk[j]
            
            if target in word2idx and all(c in word2idx for c in context):
                train_data.append((
                    [word2idx[c] for c in context],
                    word2idx[target]
                ))
        
        # Free memory
        del chunk
    
    print(f"Created {len(train_data)} training pairs")
    
    # Split into train and test sets
    np.random.shuffle(train_data)
    split_idx = int(len(train_data) * 0.9)
    train_set = train_data[:split_idx]
    test_set = train_data[split_idx:]
    
    print(f"Training set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    
    # Free memory
    del train_data
    del words
    
    # Initialize model and optimizer
    model = CBOW(len(vocab), embedding_dim=wandb.config.embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Training loop
    batch_size = wandb.config.batch_size
    num_epochs = wandb.config.epochs
    
    # Define test words
    test_words = ["computer", "technology", "data", "learning", "system"]
    test_words = [w for w in test_words if w in word2idx]
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        # Shuffle training data
        np.random.shuffle(train_set)
        
        with tqdm(range(0, len(train_set), batch_size), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for i in pbar:
                batch = train_set[i:i+batch_size]
                if not batch:
                    continue
                
                contexts, targets = zip(*batch)
                context_tensor = torch.LongTensor(contexts).to(device)
                target_tensor = torch.LongTensor(targets).to(device)
                
                optimizer.zero_grad()
                
                # Use mixed precision training
                with autocast():
                    outputs = model(context_tensor)
                    loss = F.cross_entropy(outputs, target_tensor)
                
                # Scale gradients and optimize
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Update metrics
                _, predicted = torch.max(outputs.data, 1)
                total += target_tensor.size(0)
                correct += (predicted == target_tensor).sum().item()
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{total_loss/(i/batch_size + 1):.2f}",
                    "acc": f"{100*correct/total:.1f}%"
                })
                
                # Free memory
                del context_tensor
                del target_tensor
                del outputs
        
        # Evaluation
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for i in range(0, len(test_set), batch_size):
                batch = test_set[i:i+batch_size]
                if not batch:
                    continue
                
                contexts, targets = zip(*batch)
                context_tensor = torch.LongTensor(contexts).to(device)
                target_tensor = torch.LongTensor(targets).to(device)
                
                outputs = model(context_tensor)
                loss = F.cross_entropy(outputs, target_tensor)
                
                _, predicted = torch.max(outputs.data, 1)
                test_total += target_tensor.size(0)
                test_correct += (predicted == target_tensor).sum().item()
                test_loss += loss.item()
        
        # Calculate metrics
        train_loss = total_loss / (len(train_set) // batch_size)
        test_loss = test_loss / (len(test_set) // batch_size)
        train_acc = 100 * correct / total
        test_acc = 100 * test_correct / test_total
        
        # Log metrics
        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "epoch": epoch
        })
        
        # Print results
        print(f"\nEpoch {epoch+1} completed")
        print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.2f}%")
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.2f}%")
        
        # Print similar words
        print("\nSimilar words:")
        for word in test_words:
            similar = get_similar_words(model, word, word2idx, idx2word)
            if isinstance(similar, list):
                print(f"{word}: {', '.join([f'{w} ({s:.3f})' for w, s in similar])}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
        }, f"data/word2vec/checkpoint_epoch_{epoch}.pt")
        
        # Clear CUDA cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # Save final embeddings
    embeddings = model.embeddings.weight.data.cpu().numpy()
    torch.save({
        'embeddings': embeddings,
        'vocab': vocab,
        'word2idx': word2idx,
        'idx2word': idx2word
    }, "data/word2vec/word2vec_cbow_final.pt")
    
    # Save in numpy format
    np.save("data/word2vec/embeddings.npy", embeddings)
    with open("data/word2vec/vocab.txt", "w") as f:
        for word in vocab:
            f.write(word + "\n")
    
    wandb.finish()

if __name__ == "__main__":
    train()
