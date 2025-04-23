import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from collections import Counter
import numpy as np
from tqdm import tqdm
import wandb
import os
import re

def clean_word(word):
    """Clean a word by removing punctuation and converting to lowercase"""
    # Remove punctuation and special characters
    word = re.sub(r'[^\w\s]', '', word)
    # Convert to lowercase
    word = word.lower()
    # Remove any remaining whitespace
    word = word.strip()
    return word

def get_combined_words():
    """Loads text8 + MS-MARCO passages into a single word list"""
    # 1. Load text8
    with open("data/text8", "r") as f:
        text8_words = [clean_word(word) for word in f.read().split()]
        text8_words = [word for word in text8_words if word]  # Remove empty strings
    
    # 2. Load MS-MARCO passages
    print("Loading MS-MARCO passages...")
    dataset = load_dataset("ms_marco", "v1.1", split="train[:100%]")
    msmarco_words = []
    for example in tqdm(dataset, desc="Processing MS-MARCO"):
        passages = example['passages']['passage_text']
        for passage in passages:
            # Clean and filter words
            words = [clean_word(word) for word in passage.split()]
            words = [word for word in words if word]  # Remove empty strings
            msmarco_words.extend(words)
    
    return text8_words + msmarco_words

def get_similar_words(model, word, word2idx, idx2word, top_k=5):
    """Get similar words using cosine similarity"""
    if word not in word2idx:
        return f"Word '{word}' not in vocabulary"
    
    word_idx = word2idx[word]
    word_embedding = model.embeddings.weight[word_idx]
    
    # Calculate cosine similarity with all words
    similarities = torch.nn.functional.cosine_similarity(
        word_embedding.unsqueeze(0),
        model.embeddings.weight,
        dim=1
    )
    
    # Get top k similar words (excluding the word itself)
    top_k_indices = torch.topk(similarities, k=top_k+1)[1][1:]  # +1 to exclude the word itself
    similar_words = [idx2word[idx.item()] for idx in top_k_indices]
    
    return similar_words

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context):
        embeds = self.embeddings(context).mean(dim=1)
        return self.linear(embeds)

def train():
    # Initialize wandb
    wandb.init(project="word2vec-cbow", config={
        "architecture": "CBOW",
        "dataset": "text8+MS-MARCO",
        "embedding_dim": 100,
        "window_size": 2,
        "batch_size": 1024,
        "test_size": 0.1
    })
    
    # Create data directory if it doesn't exist
    os.makedirs("data/word2vec", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Get all words
    print("Loading and combining datasets...")
    words = get_combined_words()
    
    # 2. Build vocabulary
    print("Building vocabulary...")
    word_counts = Counter(words)
    vocab = [word for word, count in word_counts.items() if count >= 5]
    print(f"Vocabulary size: {len(vocab)}")
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    # 3. Prepare CBOW training data
    print("Preparing training data...")
    window_size = 2
    train_data = []
    for i in tqdm(range(window_size, len(words)-window_size), desc="Creating training pairs"):
        context = words[i-window_size:i] + words[i+1:i+window_size+1]
        target = words[i]
        if target in word2idx and all(c in word2idx for c in context):
            train_data.append((
                [word2idx[c] for c in context],
                word2idx[target]
            ))
    
    # Split data into train and test sets
    np.random.shuffle(train_data)
    split_idx = int(len(train_data) * 0.9)  # 90% train, 10% test
    train_set = train_data[:split_idx]
    test_set = train_data[split_idx:]
    
    # 4. Initialize model
    model = CBOW(len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training loop with progress bar
    batch_size = 1024
    num_epochs = 5
    
    for epoch in range(num_epochs):
        model.train()
        np.random.shuffle(train_set)
        total_loss = 0
        correct = 0
        total = 0
        
        # Training phase
        with tqdm(range(0, len(train_set), batch_size)) as pbar:
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs} - Training")
            
            for i in pbar:
                batch = train_set[i:i+batch_size]
                if not batch:
                    continue
                    
                contexts, targets = zip(*batch)
                context_tensor = torch.LongTensor(contexts).to(device)
                target_tensor = torch.LongTensor(targets).to(device)
                
                optimizer.zero_grad()
                outputs = model(context_tensor)
                loss = criterion(outputs, target_tensor)
                loss.backward()
                optimizer.step()
                
                _, predicted = torch.max(outputs.data, 1)
                total += target_tensor.size(0)
                correct += (predicted == target_tensor).sum().item()
                total_loss += loss.item()
                
                pbar.set_postfix({
                    "loss": f"{total_loss/(i/batch_size + 1):.2f}",
                    "acc": f"{100*correct/total:.1f}%"
                })
        
        # Evaluation phase
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
                loss = criterion(outputs, target_tensor)
                
                _, predicted = torch.max(outputs.data, 1)
                test_total += target_tensor.size(0)
                test_correct += (predicted == target_tensor).sum().item()
                test_loss += loss.item()
        
        # Log metrics to wandb
        wandb.log({
            "train_loss": total_loss/len(train_set),
            "train_accuracy": 100*correct/total,
            "test_loss": test_loss/len(test_set),
            "test_accuracy": 100*test_correct/test_total,
            "epoch": epoch
        })
        
        # Test word similarity
        test_word = "king"
        similar_words = get_similar_words(model, test_word, word2idx, idx2word)
        print(f"\nEpoch {epoch+1} - Similar words to '{test_word}': {similar_words}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': total_loss/len(train_set),
            'test_loss': test_loss/len(test_set)
        }, f"data/word2vec/checkpoint_epoch_{epoch}.pt")
    
    # 6. Save final embeddings
    torch.save({
        'embeddings': model.embeddings.weight.data.cpu().numpy(),
        'vocab': vocab
    }, "data/word2vec/word2vec_cbow_final.pt")
    
    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    train()