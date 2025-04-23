import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from collections import Counter
import numpy as np

def get_combined_words():
    """Loads text8 + MS-MARCO passages into a single word list"""
    # 1. Load text8 (already in your data folder)
    with open("data/text8", "r") as f:
        text8_words = f.read().split()
    
    # 2. Load MS-MARCO passages (first 50k examples)
    print("Loading MS-MARCO passages...")
    dataset = load_dataset("ms_marco", "v1.1", split="train[:50000]")
    msmarco_words = []
    for example in dataset:
        # Take first 2 passages from each example
        passages = example['passages']['passage_text'][:2]
        for passage in passages:
            msmarco_words.extend(passage.lower().split())
    
    return text8_words + msmarco_words

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context):
        embeds = self.embeddings(context).mean(dim=1)  # Average context
        return self.linear(embeds)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Get all words
    words = get_combined_words()
    
    # 2. Build vocabulary (keep frequent words)
    word_counts = Counter(words)
    vocab = [word for word, count in word_counts.items() if count >= 5]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    
    # 3. Prepare CBOW training data
    window_size = 2  # 2 words before, 2 words after
    train_data = []
    for i in range(window_size, len(words)-window_size):
        context = words[i-window_size:i] + words[i+1:i+window_size+1]
        target = words[i]
        if target in word2idx and all(c in word2idx for c in context):
            train_data.append((
                [word2idx[c] for c in context],
                word2idx[target]
            ))
    
    # 4. Initialize model
    model = CBOW(len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training loop
    batch_size = 1024
    for epoch in range(5):
        np.random.shuffle(train_data)
        total_loss = 0
        
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
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
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/(len(train_data)/batch_size):.2f}")
    
    # 6. Save embeddings
    torch.save({
        'embeddings': model.embeddings.weight.data.cpu().numpy(),
        'vocab': vocab
    }, "word2vec_cbow.pt")
    print("Training complete! Embeddings saved to word2vec_cbow.pt")

if __name__ == "__main__":
    train()