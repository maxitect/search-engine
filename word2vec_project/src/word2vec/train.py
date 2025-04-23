import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import os
import wandb

class Word2VecDataset(Dataset):
    def __init__(self, data: List[Tuple[int, int]], vocab_size: int):
        self.data = data
        self.vocab_size = vocab_size
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        center_word, context_word = self.data[idx]
        return torch.tensor(center_word), torch.tensor(context_word)

def train_word2vec(
    model: nn.Module,
    train_data: List[Tuple[int, int]],
    vocab_size: int,
    batch_size: int = 128,
    num_epochs: int = 5,
    learning_rate: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    checkpoint_dir: str = 'models/word2vec/checkpoints'
) -> None:
    """Train the Word2Vec model."""
    # Initialize wandb
    wandb.init(project="word2vec", config={
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "vocab_size": vocab_size,
        "embedding_dim": model.embeddings.embedding_dim
    })
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Create dataset and dataloader
    dataset = Word2VecDataset(train_data, vocab_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (center_words, context_words) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Move data to device
            center_words = center_words.to(device)
            context_words = context_words.to(device)
            
            # Forward pass
            outputs = model(center_words)
            loss = criterion(outputs, context_words)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Log batch loss
            if batch_idx % 100 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "epoch": epoch,
                    "batch": batch_idx
                })
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        
        # Log epoch metrics
        wandb.log({
            "epoch_loss": avg_loss,
            "epoch": epoch
        })
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, checkpoint_path)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Save final model
    final_path = os.path.join('models/word2vec/trained', 'final_model.pth')
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")
    
    # Save embeddings
    embeddings_path = os.path.join('models/word2vec/trained', 'embeddings.npy')
    model.save_embeddings(embeddings_path)
    
    wandb.finish() 