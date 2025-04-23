import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from collections import Counter
import numpy as np
from tqdm import tqdm
import wandb

def get_combined_words():
    """Loads text8 + MS-MARCO passages into a single word list"""
    # 1. Load text8
    with open("data/text8", "r") as f:
        text8_words = f.read().split()
    
    # 2. Load MS-MARCO passages
    print("Loading MS-MARCO passages...")
    dataset = load_dataset("ms_marco", "v1.1", split="train[:50000]")
    msmarco_words = []
    for example in tqdm(dataset, desc="Processing MS-MARCO"):
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
        embeds = self.embeddings(context).mean(dim=1)
        return self.linear(embeds)

def train():
    # Initialize wandb
    wandb.init(project="word2vec-cbow", config={
        "architecture": "CBOW",
        "dataset": "text8+MS-MARCO",
        "embedding_dim": 100,
        "window_size": 2,
        "batch_size": 1024
    })
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Get all words
    print("Loading and combining datasets...")
    words = get_combined_words()
    
    # 2. Build vocabulary
    print("Building vocabulary...")
    word_counts = Counter(words)
    vocab = [word for word, count in word_counts.items() if count >= 5]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    
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
    
    # 4. Initialize model
    model = CBOW(len(vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # 5. Training loop with progress bar
    batch_size = 1024
    num_epochs = 5
    
    for epoch in range(num_epochs):
        model.train()
        np.random.shuffle(train_data)
        total_loss = 0
        correct = 0
        total = 0
        
        with tqdm(range(0, len(train_data), batch_size)) as pbar:
            pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")
            
            for i in pbar:
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
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += target_tensor.size(0)
                correct += (predicted == target_tensor).sum().item()
                
                total_loss += loss.item()
                
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{total_loss/(i/batch_size + 1):.2f}",
                    "acc": f"{100*correct/total:.1f}%"
                })
                
                # Log to wandb
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_accuracy": 100*correct/total,
                    "epoch": epoch + (i/len(train_data))
                })
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss/len(train_data),
        }, f"checkpoint_epoch_{epoch}.pt")
    
    # 6. Save final embeddings
    torch.save({
        'embeddings': model.embeddings.weight.data.cpu().numpy(),
        'vocab': vocab
    }, "word2vec_cbow_final.pt")
    
    wandb.finish()
    print("Training complete!")

if __name__ == "__main__":
    train()