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

def triplet_loss_function(query_emb, pos_doc_emb, neg_doc_emb, margin=0.2):
    """Compute triplet loss with margin."""
    pos_scores = torch.sum(query_emb * pos_doc_emb, dim=1)
    neg_scores = torch.sum(query_emb * neg_doc_emb, dim=1)
    loss = torch.clamp(neg_scores - pos_scores + margin, min=0.0)
    return torch.mean(loss)

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
    
    # Create datasets
    train_dataset = CustomMSMARCODataset(config['train_path'], vocab_size=config['vocab_size'])
    val_dataset = CustomMSMARCODataset(config['val_path'], vocab_size=config['vocab_size'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Initialize model
    model = CustomTwoTowerModel(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim']
    )
    model = model.to(device)
    
    # Initialize optimizer with lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'] * 0.1)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )
    
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
    patience_counter = 0
    for epoch in range(start_epoch, config['epochs']):
        epoch_start_time = time.time()
        
        # Training
        model.train()
        train_losses = []
        batch_times = []
        grad_norms = []
        batch_count = 0
        nan_batches = 0
        
        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch in progress:
            try:
                batch_start_time = time.time()
                batch_count += 1
                
                # Move batch to device
                query = batch['query'].to(device)
                pos_passage = batch['passage'].to(device)
                neg_passage = batch['passage'][torch.randperm(len(batch['passage']))].to(device)
                
                # Forward pass
                query_emb = model.query_tower(query)
                pos_emb = model.passage_tower(pos_passage)
                neg_emb = model.passage_tower(neg_passage)
                
                # Compute triplet loss
                loss = triplet_loss_function(
                    query_emb, pos_emb, neg_emb, 
                    margin=config.get('margin', 0.2)
                )
                
                if torch.isnan(loss):
                    nan_batches += 1
                    print(f"Warning: NaN loss in batch {batch_count}")
                    continue
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Calculate gradient norms
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norms.append(total_norm ** 0.5)
                
                optimizer.step()
                
                train_losses.append(loss.item())
                batch_times.append(time.time() - batch_start_time)
                
                progress.set_postfix({'loss': loss.item()})
                
                # Log batch metrics
                if batch_count % 100 == 0:
                    wandb.log({
                        'train/batch_loss': loss.item(),
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
        val_scores = []
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    query = batch['query'].to(device)
                    pos_passage = batch['passage'].to(device)
                    neg_passage = batch['passage'][torch.randperm(len(batch['passage']))].to(device)
                    
                    # Forward pass
                    query_emb = model.query_tower(query)
                    pos_emb = model.passage_tower(pos_passage)
                    neg_emb = model.passage_tower(neg_passage)
                    
                    # Compute triplet loss
                    loss = triplet_loss_function(
                        query_emb, pos_emb, neg_emb,
                        margin=config.get('margin', 0.2)
                    )
                    
                    if not torch.isnan(loss):
                        val_losses.append(loss.item())
                        val_scores.extend(torch.sum(query_emb * pos_emb, dim=1).cpu().numpy())
                        
                except RuntimeError as e:
                    if "CUDNN_STATUS_EXECUTION_FAILED" in str(e):
                        print("CUDA error in validation, skipping batch...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    raise e
        
        # Calculate metrics
        avg_val_loss = np.mean(val_losses) if val_losses else float('nan')
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save checkpoint
        checkpoint_path = os.path.join(models_dir, f'Custom_Top_Tower_Epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'config': config
        }, checkpoint_path)
        
        # Save best model
        if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            best_model_path = os.path.join(models_dir, 'Custom_Top_Tower_Epoch.pth')
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': config
            }, best_model_path)
            
            # Create wandb artifact for best model
            best_artifact = wandb.Artifact('custom-two-tower-best', type='model')
            best_artifact.add_file(best_model_path)
            wandb.log_artifact(best_artifact)
            
            print(f"\nNew best model saved at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Log epoch metrics
        wandb.log({
            'epoch/train_loss': np.mean(train_losses) if train_losses else float('nan'),
            'epoch/val_loss': avg_val_loss,
            'epoch/time': time.time() - epoch_start_time,
            'epoch/best_val_loss': best_val_loss,
            'epoch/best_epoch': best_epoch,
            'epoch/nan_batches': nan_batches,
            'epoch/learning_rate': optimizer.param_groups[0]['lr'],
            'epoch/avg_grad_norm': np.mean(grad_norms) if grad_norms else 0.0,
            'epoch/avg_batch_time': np.mean(batch_times) if batch_times else 0.0,
            'epoch/val_scores_mean': np.mean(val_scores) if val_scores else 0.0,
            'epoch/val_scores_std': np.std(val_scores) if val_scores else 0.0
        })
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train Loss: {np.mean(train_losses) if train_losses else "nan":.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})')
        print(f'  NaN Batches: {nan_batches}')
        print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Early stopping
        if patience_counter >= config.get('patience', 5):
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    print(f'Training completed! Best validation loss: {best_val_loss:.4f}')
    wandb.finish()

if __name__ == '__main__':
    config = {
        'train_path': '/root/search-engine/data/msmarco/train.json',
        'val_path': '/root/search-engine/data/msmarco/val.json',
        'vocab_size': 100000,
        'embedding_dim': 300,
        'hidden_dim': 256,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 10,
        'margin': 0.2,
        'patience': 5
    }
    
    train_model(config)
