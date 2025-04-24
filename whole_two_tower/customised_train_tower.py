import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
import json
from tqdm import tqdm
import os
import time
import psutil
import gc

class CustomMSMARCODataset(Dataset):
    def __init__(self, data_path, max_query_len=32, max_passage_len=256, vocab_size=100000):
        self.data = []
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len
        self.vocab_size = vocab_size
        
        # Load and preprocess data
        with open(data_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                query = item['query']
                passages = item['passages']
                is_selected = item['is_selected']
                
                # Process passages
                for passage, selected in zip(passages, is_selected):
                    passage_text = passage['passage_text']
                    
                    # Clamp indices to valid range
                    query = [min(idx, vocab_size - 1) for idx in query]
                    passage_text = [min(idx, vocab_size - 1) for idx in passage_text]
                    
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

class CustomTwoTowerModel(nn.Module):
    """Two-tower model without Gensim embeddings."""
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        
        # Query tower
        self.query_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.query_rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.query_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Passage tower
        self.passage_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.passage_rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.passage_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.passage_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, query, passage):
        try:
            # Clamp indices to valid range
            query = torch.clamp(query, 0, self.query_embedding.num_embeddings - 1)
            passage = torch.clamp(passage, 0, self.passage_embedding.num_embeddings - 1)
            
            # Query tower
            query_emb = self.query_embedding(query)
            query_rnn_out, _ = self.query_rnn(query_emb)
            query_rep = self.query_projection(query_rnn_out[:, -1, :])
            
            # Passage tower
            passage_emb = self.passage_embedding(passage)
            passage_rnn_out, _ = self.passage_rnn(passage_emb)
            
            # Attention
            attention_weights = self.passage_attention(passage_rnn_out)
            attention_weights = torch.softmax(attention_weights, dim=1)
            attended = torch.sum(passage_rnn_out * attention_weights, dim=1)
            passage_rep = self.passage_projection(attended)
            
            # Similarity score
            return torch.sum(query_rep * passage_rep, dim=1)
        except RuntimeError as e:
            if "CUDNN_STATUS_EXECUTION_FAILED" in str(e):
                # Clear CUDA cache and retry
                torch.cuda.empty_cache()
                gc.collect()
                return self.forward(query, passage)
            raise e

def train_step(model, optimizer, query_batch, pos_passage_batch, neg_passage_batch):
    try:
        # Forward pass
        pos_scores = model(query_batch, pos_passage_batch)
        neg_scores = model(query_batch, neg_passage_batch)
        
        # Compute loss
        loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        return loss.item()
    except RuntimeError as e:
        if "CUDNN_STATUS_EXECUTION_FAILED" in str(e):
            # Clear CUDA cache and retry
            torch.cuda.empty_cache()
            gc.collect()
            return train_step(model, optimizer, query_batch, pos_passage_batch, neg_passage_batch)
        raise e

def evaluate(model, query_batch, passage_batch, labels):
    with torch.no_grad():
        scores = model(query_batch, passage_batch)
        predictions = (scores > 0).float()
        accuracy = (predictions == labels).float().mean()
    return accuracy.item()

def train_model(config):
    # Initialize wandb
    wandb.init(
        project="custom-two-tower-search",
        config=config,
        name=f"custom_two_tower_{time.strftime('%Y%m%d_%H%M%S')}",
        save_code=True
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Check for existing model checkpoint
    checkpoint_path = os.path.join(models_dir, 'Custom_Top_Tower_Epoch.pth')
    start_epoch = 0
    
    if os.path.exists(checkpoint_path):
        print("\nFound existing model checkpoint!")
        choice = input("Do you want to resume training from the last saved epoch? (y/n): ").lower()
        if choice == 'y':
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch']
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("Starting fresh training")
    
    # Create datasets with vocab_size
    train_dataset = CustomMSMARCODataset(config['train_path'], vocab_size=config['vocab_size'])
    val_dataset = CustomMSMARCODataset(config['val_path'], vocab_size=config['vocab_size'])
    test_dataset = CustomMSMARCODataset(config['test_path'], vocab_size=config['vocab_size'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Initialize model
    model = CustomTwoTowerModel(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim']
    )
    model = model.to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Load model and optimizer states if resuming
    if start_epoch > 0:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['val_loss']
        best_epoch = checkpoint['epoch']
    else:
        best_val_loss = float('inf')
        best_epoch = 0
    
    # Training loop
    for epoch in range(start_epoch, config['epochs']):
        epoch_start_time = time.time()
        
        # Training
        model.train()
        train_losses = []
        batch_times = []
        grad_norms = []
        batch_count = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            try:
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
                
                # Log metrics
                if batch_count % 100 == 0:
                    wandb.log({
                        'train/batch_loss': loss,
                        'train/grad_norm': total_norm ** 0.5,
                        'train/batch_time': time.time() - batch_start_time,
                        'train/learning_rate': optimizer.param_groups[0]['lr'],
                        'train/batch': batch_count,
                        'train/epoch': epoch + 1
                    })
            except RuntimeError as e:
                if "CUDNN_STATUS_EXECUTION_FAILED" in str(e):
                    print(f"CUDA error in batch {batch_count}, retrying...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise e
        
        # Validation
        model.eval()
        val_losses = []
        val_accuracies = []
        
        for batch in val_loader:
            try:
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
            except RuntimeError as e:
                if "CUDNN_STATUS_EXECUTION_FAILED" in str(e):
                    print("CUDA error in validation, skipping batch...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise e
        
        # Calculate metrics
        avg_val_loss = np.mean(val_losses)
        avg_val_accuracy = np.mean(val_accuracies)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            
            model_path = os.path.join(models_dir, 'Custom_Top_Tower_Epoch.pth')
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': avg_val_accuracy,
                'config': config
            }, model_path)
            print(f"\nNew best model saved at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
        
        # Log epoch metrics
        wandb.log({
            'epoch/train_loss': np.mean(train_losses),
            'epoch/val_loss': avg_val_loss,
            'epoch/val_accuracy': avg_val_accuracy,
            'epoch/time': time.time() - epoch_start_time,
            'epoch/best_val_loss': best_val_loss,
            'epoch/best_epoch': best_epoch
        })
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train Loss: {np.mean(train_losses):.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val Accuracy: {avg_val_accuracy:.4f}')
        print(f'  Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})')
        
        # Clear CUDA cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    wandb.finish()

if __name__ == '__main__':
    config = {
        'train_path': '/root/search-engine/data/msmarco/train.json',
        'val_path': '/root/search-engine/data/msmarco/val.json',
        'test_path': '/root/search-engine/data/msmarco/test.json',
        'vocab_size': 100000,  # Adjust based on your vocabulary size
        'embedding_dim': 300,
        'hidden_dim': 256,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 10
    }
    
    train_model(config)
