import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from two_tower_model import TwoTowerModel, train_step, evaluate
import wandb
import json
from tqdm import tqdm
import os

class MSMARCODataset(Dataset):
    def __init__(self, data_path, max_query_len=32, max_passage_len=256):
        self.data = []
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len
        
        # Load and preprocess data
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                query = item['query']
                passages = item['passages']
                is_selected = item['is_selected']
                
                # Pad or truncate query
                query = query[:max_query_len]
                query = query + [0] * (max_query_len - len(query))
                
                # Process passages
                for passage, selected in zip(passages, is_selected):
                    passage_text = passage['passage_text']
                    # Pad or truncate passage
                    passage_text = passage_text[:max_passage_len]
                    passage_text = passage_text + [0] * (max_passage_len - len(passage_text))
                    
                    self.data.append({
                        'query': query,
                        'passage': passage_text,
                        'is_selected': selected
                    })
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'query': torch.LongTensor(item['query']),
            'passage': torch.LongTensor(item['passage']),
            'is_selected': torch.FloatTensor([item['is_selected']])
        }

def create_negative_samples(batch):
    """Create negative samples by shuffling passages."""
    batch_size = len(batch['query'])
    neg_indices = torch.randperm(batch_size)
    return {
        'query': batch['query'],
        'passage': batch['passage'][neg_indices],
        'is_selected': torch.zeros_like(batch['is_selected'])
    }

def train_model(config):
    # Initialize wandb
    wandb.init(project="two-tower-search", config=config)
    
    # Create datasets
    train_dataset = MSMARCODataset(config['train_path'])
    val_dataset = MSMARCODataset(config['val_path'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Initialize model
    model = TwoTowerModel(
        gensim_model_path=config['gensim_model_path'],
        hidden_dim=config['hidden_dim']
    )
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    best_val_accuracy = 0
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            # Move batch to device
            query = batch['query'].to(device)
            pos_passage = batch['passage'].to(device)
            
            # Create negative samples
            neg_batch = create_negative_samples(batch)
            neg_passage = neg_batch['passage'].to(device)
            
            # Training step
            loss = train_step(model, optimizer, query, pos_passage, neg_passage)
            train_losses.append(loss)
        
        # Validation
        model.eval()
        val_accuracies = []
        for batch in val_loader:
            query = batch['query'].to(device)
            passage = batch['passage'].to(device)
            is_selected = batch['is_selected'].to(device)
            
            accuracy = evaluate(model, query, passage, is_selected)
            val_accuracies.append(accuracy)
        
        # Log metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_accuracy = np.mean(val_accuracies)
        
        wandb.log({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_accuracy': avg_val_accuracy
        })
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Accuracy: {avg_val_accuracy:.4f}')
        
        # Save best model
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')
    
    wandb.finish()

if __name__ == '__main__':
    config = {
        'train_path': '/root/search-engine/data/msmarco/train.json',
        'val_path': '/root/search-engine/data/msmarco/val.json',
        'gensim_model_path': '/root/search-engine/models/text8_embeddings/word2vec_model',
        'hidden_dim': 256,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 10
    }
    
    # Add debug logging
    print("Checking data files...")
    print(f"Train path exists: {os.path.exists(config['train_path'])}")
    print(f"Val path exists: {os.path.exists(config['val_path'])}")
    
    if os.path.exists(config['train_path']):
        with open(config['train_path'], 'r') as f:
            lines = list(f)
            print(f"Train file has {len(lines)} lines")
    
    if os.path.exists(config['val_path']):
        with open(config['val_path'], 'r') as f:
            lines = list(f)
            print(f"Val file has {len(lines)} lines")
    
    train_model(config) 