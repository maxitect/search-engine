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
import time
import psutil
import gc

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
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create datasets
    train_dataset = MSMARCODataset(config['train_path'])
    val_dataset = MSMARCODataset(config['val_path'])
    test_dataset = MSMARCODataset(config['test_path'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Calculate logging intervals
    train_batches_per_epoch = len(train_loader)
    log_interval = max(1, train_batches_per_epoch // 10)  # Log 10 times per epoch
    test_interval = max(1, train_batches_per_epoch // 5)  # Test 5 times per epoch
    
    # Initialize model
    model = TwoTowerModel(
        gensim_model_path=config['gensim_model_path'],
        hidden_dim=config['hidden_dim']
    )
    
    # Print model architecture
    print("\nModel Architecture:")
    print("Query Tower:")
    print(f"- BiGRU: {config['hidden_dim']} hidden units, 2 layers")
    print(f"- Projection: {config['hidden_dim']*2} -> {config['hidden_dim']}")
    print("\nPassage Tower:")
    print(f"- BiGRU: {config['hidden_dim']} hidden units, 2 layers")
    print(f"- Attention: {config['hidden_dim']*2} -> {config['hidden_dim']} -> 1")
    print(f"- Projection: {config['hidden_dim']*2} -> {config['hidden_dim']}")
    
    # Move model to device
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(config['epochs']):
        epoch_start_time = time.time()
        
        # Training
        model.train()
        train_losses = []
        batch_times = []
        grad_norms = []
        batch_count = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            batch_start_time = time.time()
            batch_count += 1
            
            # Move batch to device
            query = batch['query'].to(device)
            pos_passage = batch['passage'].to(device)
            
            # Create negative samples
            neg_batch = create_negative_samples(batch)
            neg_passage = neg_batch['passage'].to(device)
            
            # Training step
            loss = train_step(model, optimizer, query, pos_passage, neg_passage)
            train_losses.append(loss)
            
            # Calculate gradient norms
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norms.append(total_norm ** 0.5)
            
            batch_times.append(time.time() - batch_start_time)
            
            # Log intermediate training metrics
            if batch_count % log_interval == 0:
                avg_train_loss = np.mean(train_losses[-log_interval:])
                avg_grad_norm = np.mean(grad_norms[-log_interval:])
                avg_batch_time = np.mean(batch_times[-log_interval:])
                
                wandb.log({
                    'epoch': epoch + 1,
                    'batch': batch_count,
                    'train_loss': avg_train_loss,
                    'grad_norm': avg_grad_norm,
                    'batch_time': avg_batch_time,
                    'learning_rate': optimizer.param_groups[0]['lr']
                })
            
            # Run intermediate test
            if batch_count % test_interval == 0:
                model.eval()
                test_losses = []
                for test_batch in test_loader:
                    query = test_batch['query'].to(device)
                    passage = test_batch['passage'].to(device)
                    is_selected = test_batch['is_selected'].to(device)
                    
                    with torch.no_grad():
                        pos_scores = model(query, passage)
                        neg_scores = model(query, passage[torch.randperm(len(passage))])
                        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
                        test_losses.append(loss.item())
                
                avg_test_loss = np.mean(test_losses)
                wandb.log({
                    'epoch': epoch + 1,
                    'batch': batch_count,
                    'test_loss': avg_test_loss
                })
                model.train()
        
        # Validation
        model.eval()
        val_losses = []
        val_accuracies = []
        for batch in val_loader:
            query = batch['query'].to(device)
            passage = batch['passage'].to(device)
            is_selected = batch['is_selected'].to(device)
            
            with torch.no_grad():
                pos_scores = model(query, passage)
                neg_scores = model(query, passage[torch.randperm(len(passage))])
                loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
                val_losses.append(loss.item())
            
            accuracy = evaluate(model, query, passage, is_selected)
            val_accuracies.append(accuracy)
        
        # Calculate validation metrics
        avg_val_loss = np.mean(val_losses)
        avg_val_accuracy = np.mean(val_accuracies)
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            
            # Save the best model
            model_path = os.path.join(models_dir, 'Top_Tower_Epoch.pth')
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': avg_val_accuracy,
                'config': config
            }, model_path)
            print(f"\nNew best model saved at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
        
        # Test
        test_accuracies = []
        for batch in test_loader:
            query = batch['query'].to(device)
            passage = batch['passage'].to(device)
            is_selected = batch['is_selected'].to(device)
            
            accuracy = evaluate(model, query, passage, is_selected)
            test_accuracies.append(accuracy)
        
        # Calculate metrics
        avg_train_loss = np.mean(train_losses)
        avg_test_accuracy = np.mean(test_accuracies)
        avg_grad_norm = np.mean(grad_norms)
        avg_batch_time = np.mean(batch_times)
        epoch_time = time.time() - epoch_start_time
        
        # Calculate model parameter statistics
        param_stats = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_stats[f'param_mean_{name}'] = param.data.mean().item()
                param_stats[f'param_std_{name}'] = param.data.std().item()
        
        # Get memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
            memory_cached = torch.cuda.memory_reserved() / 1024**2  # MB
        else:
            process = psutil.Process()
            memory_allocated = process.memory_info().rss / 1024**2  # MB
        
        # Log metrics
        log_dict = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_accuracy': avg_val_accuracy,
            'test_accuracy': avg_test_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr'],
            'avg_grad_norm': avg_grad_norm,
            'avg_batch_time': avg_batch_time,
            'epoch_time': epoch_time,
            'memory_allocated': memory_allocated,
        }
        
        if torch.cuda.is_available():
            log_dict['memory_cached'] = memory_cached
        
        # Add parameter statistics
        log_dict.update(param_stats)
        
        wandb.log(log_dict)
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train Loss: {avg_train_loss:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val Accuracy: {avg_val_accuracy:.4f}')
        print(f'  Test Accuracy: {avg_test_accuracy:.4f}')
        print(f'  Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})')
        print(f'  Epoch Time: {epoch_time:.2f}s')
    
    wandb.finish()
    
    # Print final summary
    print("\nTraining Summary:")
    print(f"Best model saved at epoch {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {os.path.join(models_dir, 'Top_Tower_Epoch.pth')}")

if __name__ == '__main__':
    config = {
        'train_path': '/root/search-engine/data/msmarco/train.json',
        'val_path': '/root/search-engine/data/msmarco/val.json',
        'test_path': '/root/search-engine/data/msmarco/test.json',
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
    print(f"Test path exists: {os.path.exists(config['test_path'])}")
    
    if os.path.exists(config['train_path']):
        with open(config['train_path'], 'r') as f:
            lines = list(f)
            print(f"Train file has {len(lines)} lines")
    
    if os.path.exists(config['val_path']):
        with open(config['val_path'], 'r') as f:
            lines = list(f)
            print(f"Val file has {len(lines)} lines")
    
    if os.path.exists(config['test_path']):
        with open(config['test_path'], 'r') as f:
            lines = list(f)
            print(f"Test file has {len(lines)} lines")
    
    train_model(config) 