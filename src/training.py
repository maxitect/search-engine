# src/training.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
import wandb
from datetime import datetime

models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
os.makedirs(models_dir, exist_ok=True)

model_path = os.path.join(models_dir, 'best_model.pth')


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
        
    def forward(self, query, pos, neg):
        pos_dist = torch.sum((query - pos)**2, dim=1)
        neg_dist = torch.sum((query - neg)**2, dim=1)
        losses = torch.relu(pos_dist - neg_dist + self.margin)
        return losses.mean()



def train_model(model, train_loader, val_loader, num_epochs=3, checkpoint_dir='checkpoints'):
    """Train the model with wandb logging and checkpointing"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize wandb
    wandb.init(
        project="search-engine",
        name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "learning_rate": 1e-4,
            "architecture": "TwoTowerModel",
            "dataset": "MS MARCO",
            "epochs": num_epochs,
        }
    )
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Loss function and optimizer
    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Add progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for batch_idx, (query, pos, neg) in enumerate(train_pbar):
            # Move data to device
            query = {k: v.to(device) for k, v in query.items()}
            pos = {k: v.to(device) for k, v in pos.items()}
            neg = {k: v.to(device) for k, v in neg.items()}
            
            # Forward pass
            query_repr, pos_repr = model(query, pos)
            _, neg_repr = model(query, neg)
            
            # Compute loss
            loss = criterion(query_repr, pos_repr, neg_repr)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item()})
            
            # Log batch loss
            if batch_idx % 10 == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/batch": batch_idx + epoch * len(train_loader)
                })
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        # Add progress bar for validation
        val_pbar = tqdm(val_loader, desc="Validating")
        with torch.no_grad():
            for query, pos, neg in val_pbar:
                query = {k: v.to(device) for k, v in query.items()}
                pos = {k: v.to(device) for k, v in pos.items()}
                neg = {k: v.to(device) for k, v in neg.items()}
                
                query_repr, pos_repr = model(query, pos)
                _, neg_repr = model(query, neg)
                
                loss = criterion(query_repr, pos_repr, neg_repr)
                val_loss += loss.item()
                
                # Update validation progress bar
                val_pbar.set_postfix({'val_loss': loss.item()})
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Log epoch metrics
        wandb.log({
            "train/epoch_loss": avg_train_loss,
            "val/epoch_loss": avg_val_loss,
            "epoch": epoch
        })
        
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"✅ New best model found at epoch {epoch + 1}, saving to {os.path.join(checkpoint_dir, 'best_model.pth')}")
    
    # Finish wandb run
    wandb.finish()
    
    return model

def collate_batch(batch):
    """Collate function for the DataLoader"""
    queries = [item[0] for item in batch]
    positives = [item[1] for item in batch]
    negatives = [item[2] for item in batch]
    
    # Stack the input_ids and attention_masks
    query_inputs = {
        'input_ids': torch.stack([q['input_ids'].squeeze(0) for q in queries]),
        'attention_mask': torch.stack([q['attention_mask'].squeeze(0) for q in queries])
    }
    
    pos_inputs = {
        'input_ids': torch.stack([p['input_ids'].squeeze(0) for p in positives]),
        'attention_mask': torch.stack([p['attention_mask'].squeeze(0) for p in positives])
    }
    
    neg_inputs = {
        'input_ids': torch.stack([n['input_ids'].squeeze(0) for n in negatives]),
        'attention_mask': torch.stack([n['attention_mask'].squeeze(0) for n in negatives])
    }
    
    return query_inputs, pos_inputs, neg_inputs
