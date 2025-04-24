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
from gensim.models import Word2Vec
import pickle

class CustomMSMARCODataset(Dataset):
    def __init__(self, data_path, max_query_len=32, max_passage_len=256, vocab_size=100000, data_fraction=10, use_bert=False):
        self.data = []
        self.text_data = []  # Store original text
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len
        self.vocab_size = vocab_size
        self.use_bert = use_bert
        
        # Load embeddings based on choice
        if use_bert:
            with open('/root/search-engine/models/text8_embeddings/inherited_bert_embeddings.pkl', 'rb') as f:
                self.embeddings = pickle.load(f)
            with open('/root/search-engine/models/text8_embeddings/inherited_bert_vocab.json', 'r') as f:
                self.vocab = json.load(f)
        else:
            self.word2vec_model = Word2Vec.load('/root/search-engine/models/text8_embeddings/word2vec_model')
            self.vocab = {word: idx for idx, word in enumerate(self.word2vec_model.wv.index_to_key)}
        
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
                    
                    # Store the actual text content
                    self.text_data.append({
                        'query': item.get('query_text', 'No query text available'),
                        'passage': passage.get('passage_text_original', 'No passage text available'),
                        'is_selected': selected
                    })
        
        # Use configurable fraction of the data
        fraction = 1.0 / data_fraction
        self.data = self.data[:int(len(self.data) * fraction)]
        self.text_data = self.text_data[:int(len(self.text_data) * fraction)]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'query': torch.LongTensor(item['query']),
            'passage': torch.LongTensor(item['passage']),
            'is_selected': torch.FloatTensor([item['is_selected']])
        }

    def get_text(self, idx):
        """Get the tokenized data for a given index."""
        return self.text_data[idx]

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
        
        # Query tower components
        self.query_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.query_gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.query_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Passage tower components
        self.passage_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.passage_gru = nn.GRU(
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
    
    def query_tower(self, query):
        """Process query through query tower."""
        query = torch.clamp(query, 0, self.query_embedding.num_embeddings - 1)
        query_emb = self.query_embedding(query)
        query_rnn_out, _ = self.query_gru(query_emb)
        query_rep = self.query_projection(query_rnn_out[:, -1, :])
        return query_rep
    
    def passage_tower(self, passage):
        """Process passage through passage tower."""
        passage = torch.clamp(passage, 0, self.passage_embedding.num_embeddings - 1)
        passage_emb = self.passage_embedding(passage)
        passage_rnn_out, _ = self.passage_gru(passage_emb)
        
        # Attention
        attention_weights = self.passage_attention(passage_rnn_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        attended = torch.sum(passage_rnn_out * attention_weights, dim=1)
        passage_rep = self.passage_projection(attended)
        return passage_rep
    
    def forward(self, query, passage):
        try:
            # Get embeddings from both towers
            query_rep = self.query_tower(query)
            passage_rep = self.passage_tower(passage)
            
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

def evaluate(model, query_batch, passage_batch, labels):
    with torch.no_grad():
        query_emb = model.query_tower(query_batch)
        passage_emb = model.passage_tower(passage_batch)
        scores = torch.sum(query_emb * passage_emb, dim=1)
        predictions = (scores > 0).float()
        accuracy = (predictions == labels).float().mean()
        return accuracy.item(), scores

def get_top_examples(model, query_batch, passage_batch, dataset, k=5):
    """Get top k examples with highest cosine similarity."""
    with torch.no_grad():
        query_emb = model.query_tower(query_batch)
        passage_emb = model.passage_tower(passage_batch)
        scores = torch.sum(query_emb * passage_emb, dim=1)
        
        # Ensure k is not larger than the batch size
        k = min(k, len(scores))
        if k == 0:
            return torch.tensor([]), torch.tensor([])
            
        top_scores, top_indices = torch.topk(scores, k)
        return top_scores.cpu().numpy(), top_indices.cpu().numpy()

def load_vocabulary(vocab_path):
    """Load vocabulary mapping from file."""
    vocab = {}
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, idx = line.strip().split()
            vocab[int(idx)] = word
    return vocab

def load_word2vec_vocab(model_path):
    """Load vocabulary from a trained Word2Vec model."""
    model = Word2Vec.load(model_path)
    return model.wv

def tokens_to_text(tokens, word_vectors):
    """Convert token IDs to text using either Word2Vec or BERT vocabulary."""
    words = []
    for token in tokens:
        if token == 0:  # Skip padding tokens
            continue
        try:
            # Handle both Word2Vec and BERT vocabularies
            if hasattr(word_vectors, 'index_to_key'):  # Word2Vec case
                word = word_vectors.index_to_key[token]
            else:  # BERT case (list of words)
                word = word_vectors[token]
            words.append(word)
        except (IndexError, KeyError):
            words.append(f'[UNK_{token}]')
    return ' '.join(words)

def print_top_example(model, query_batch, passage_batch, dataset, batch_idx, top_idx, word_vectors):
    """Print a formatted example showing the actual query and passage text."""
    # Get the actual data from the dataset
    item = dataset.get_text(batch_idx)
    print("\nTop Similarity Example:")
    print("Query:", tokens_to_text(query_batch[top_idx].cpu().numpy(), word_vectors))
    print("Passage:", tokens_to_text(passage_batch[top_idx].cpu().numpy(), word_vectors))
    print("Is selected:", item['is_selected'])

def show_random_example(model, val_loader, val_dataset, device, word_vectors):
    """Show a random query and its most similar passage."""
    # Get a random batch
    batch = next(iter(val_loader))
    query = batch['query'].to(device)
    passage = batch['passage'].to(device)
    is_selected = batch['is_selected'].to(device)
    
    # Get model predictions
    with torch.no_grad():
        query_emb = model.query_tower(query)
        passage_emb = model.passage_tower(passage)
        scores = torch.sum(query_emb * passage_emb, dim=1)
        
        # Get the most similar passage
        top_score, top_idx = torch.max(scores, dim=0)
        
        # Print the example
        print("\nRandom Example:")
        print("Query:", tokens_to_text(query[0].cpu().numpy(), word_vectors))
        print("Most Similar Passage:", tokens_to_text(passage[top_idx].cpu().numpy(), word_vectors))
        print(f"Similarity Score: {top_score.item():.4f}")
        print(f"Correct Match: {'Yes' if is_selected[top_idx].item() == 1 else 'No'}")
        print(f"Batch Size: {len(query)} examples")

def evaluate_test_set(model, test_loader, device, num_batches=10):
    """Evaluate model on a subset of test batches."""
    model.eval()
    test_losses = []
    
    with torch.no_grad():
        for i, test_batch in enumerate(test_loader):
            if i >= num_batches:  # Only evaluate on first num_batches
                break
                
            query = test_batch['query'].to(device)
            pos_passage = test_batch['passage'].to(device)
            neg_passage = test_batch['passage'][torch.randperm(len(test_batch['passage']))].to(device)
            
            query_emb = model.query_tower(query)
            pos_emb = model.passage_tower(pos_passage)
            neg_emb = model.passage_tower(neg_passage)
            
            loss = triplet_loss_function(
                query_emb, pos_emb, neg_emb,
                margin=0.2
            )
            test_losses.append(loss.item())
    
    model.train()
    return np.mean(test_losses) if test_losses else float('nan')

def train_model(config):
    # Initialize wandb
    wandb.init(
        project="custom-two-tower-search",
        config=config,
        name=f"small_custom_two_tower_{time.strftime('%Y%m%d_%H%M%S')}",
        save_code=True
    )
    
    # Ask user for embedding choice
    print("\nChoose embedding type:")
    print("1. Word2Vec embeddings (custom trained)")
    print("2. BERT embeddings (pre-trained)")
    choice = input("Enter your choice (1 or 2): ").strip()
    
    use_bert = choice == "2"
    if use_bert:
        print("Using pre-trained BERT embeddings")
        # Load BERT embeddings to verify they exist
        try:
            with open('/root/search-engine/models/text8_embeddings/inherited_bert_embeddings.pkl', 'rb') as f:
                bert_embeddings = pickle.load(f)
            print(f"Loaded BERT embeddings with {len(bert_embeddings)} words")
        except FileNotFoundError:
            print("Error: BERT embeddings not found. Please run Inherit_bert.py first.")
            return
    else:
        print("Using Word2Vec embeddings")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Create datasets with embedding choice
    train_dataset = CustomMSMARCODataset(
        config['train_path'], 
        vocab_size=config['vocab_size'], 
        data_fraction=config['data_fraction'],
        use_bert=use_bert
    )
    val_dataset = CustomMSMARCODataset(
        config['val_path'], 
        vocab_size=config['vocab_size'], 
        data_fraction=config['data_fraction'],
        use_bert=use_bert
    )
    test_dataset = CustomMSMARCODataset(
        config['test_path'], 
        vocab_size=config['vocab_size'], 
        data_fraction=config['data_fraction'],
        use_bert=use_bert
    )
    
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
    
    # Initialize optimizer with lower learning rate
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'] * 0.1)
    
    # Learning rate scheduler: use cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=100  # Number of epochs for one cycle
    )
    
    # Check for existing model checkpoint
    checkpoint_path = os.path.join(models_dir, 'cus_small_top_tower_epoch.pth')
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = 0
    
    if os.path.exists(checkpoint_path):
        print("\nFound existing model checkpoint!")
        choice = input("Do you want to resume training from the last saved epoch? (y/n): ").lower()
        if choice == 'y':
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint['val_loss']
            best_epoch = checkpoint['epoch']
            print(f"Resuming training from epoch {start_epoch}")
        else:
            print("Starting fresh training")
            # Clear any existing checkpoints
            for file in os.listdir(models_dir):
                if file.startswith('cus_small_top_tower_epoch'):
                    os.remove(os.path.join(models_dir, file))
    
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
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                raise e
        
        # Validation
        model.eval()
        val_losses = []
        val_accuracies = []
        val_scores = []
        top_examples = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    query = batch['query'].to(device)
                    pos_passage = batch['passage'].to(device)
                    neg_passage = batch['passage'][torch.randperm(len(batch['passage']))].to(device)
                    is_selected = batch['is_selected'].to(device)
                    
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
                    
                    # Calculate accuracy
                    accuracy, scores = evaluate(model, query, pos_passage, is_selected)
                    val_accuracies.append(accuracy)
                    
                    # Get top examples
                    top_scores, top_indices = get_top_examples(model, query, pos_passage, val_dataset)
                    top_examples.append((top_scores, top_indices))
                    
                except RuntimeError as e:
                    if "CUDNN_STATUS_EXECUTION_FAILED" in str(e):
                        print("CUDA error in validation, skipping batch...")
                        torch.cuda.empty_cache()
                        gc.collect()
                        continue
                    raise e
        
        # Calculate metrics
        avg_val_loss = np.mean(val_losses) if val_losses else float('nan')
        avg_val_accuracy = np.mean(val_accuracies) if val_accuracies else 0.0
        
        # Get top examples across all batches
        all_top_scores = np.concatenate([scores for scores, _ in top_examples if len(scores) > 0])
        all_top_indices = np.concatenate([indices for _, indices in top_examples if len(indices) > 0])
        
        if len(all_top_scores) > 0:
            # Get a random batch and its top example
            random_batch_idx = np.random.randint(0, len(top_examples))
            top_scores, top_indices = top_examples[random_batch_idx]
            
            if len(top_indices) > 0:
                # Get the appropriate word vectors based on embedding type
                if use_bert:
                    with open('/root/search-engine/models/text8_embeddings/inherited_bert_vocab.json', 'r') as f:
                        word_vectors = json.load(f)
                else:
                    word_vectors = Word2Vec.load('/root/search-engine/models/text8_embeddings/word2vec_model').wv
                
                # Get the batch data
                batch = next(iter(val_loader))
                query = batch['query'].to(device)
                passage = batch['passage'].to(device)
                
                print_top_example(model, query, passage, val_dataset, random_batch_idx, top_indices[0], word_vectors)
                print(f"Similarity Score: {top_scores[0]:.4f}\n")
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning rate: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(models_dir, f'cus_small_top_tower_epoch_{epoch+1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss,
            'val_accuracy': avg_val_accuracy,
            'config': config
        }, checkpoint_path)
        
        # Save best model
        if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            
            best_model_path = os.path.join(models_dir, 'cus_small_top_tower_epoch.pth')
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_accuracy': avg_val_accuracy,
                'config': config
            }, best_model_path)
            
            # Create wandb artifact for best model
            best_artifact = wandb.Artifact('small-custom-two-tower-best', type='model')
            best_artifact.add_file(best_model_path)
            wandb.log_artifact(best_artifact)
            
            print(f"\nNew best model saved at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Log epoch metrics
        avg_test_loss = np.mean(val_scores) if val_scores else 0.0
        wandb.log({
            'epoch/train_loss': np.mean(train_losses) if train_losses else float('nan'),
            'epoch/val_loss': avg_val_loss,
            'epoch/val_accuracy': avg_val_accuracy,
            'epoch/test_loss': avg_test_loss,  # Added test loss per epoch
            'epoch/time': time.time() - epoch_start_time,
            'epoch/best_val_loss': best_val_loss,
            'epoch/best_epoch': best_epoch,
            'epoch/nan_batches': nan_batches,
            'epoch/learning_rate': current_lr,
            'epoch/avg_grad_norm': np.mean(grad_norms) if grad_norms else 0.0,
            'epoch/avg_batch_time': np.mean(batch_times) if batch_times else 0.0,
            'epoch/val_scores_mean': avg_test_loss,
            'epoch/val_scores_std': np.std(val_scores) if val_scores else 0.0,
            'epoch/top_similarity_1': top_scores[0] if len(top_scores) > 0 else 0.0,
            'epoch/top_similarity_2': top_scores[1] if len(top_scores) > 1 else 0.0,
            'epoch/top_similarity_3': top_scores[2] if len(top_scores) > 2 else 0.0,
            'epoch/top_similarity_4': top_scores[3] if len(top_scores) > 3 else 0.0,
            'epoch/top_similarity_5': top_scores[4] if len(top_scores) > 4 else 0.0,
            'epoch/val_scores_min': np.min(val_scores) if val_scores else 0.0,
            'epoch/val_scores_max': np.max(val_scores) if val_scores else 0.0,
            'epoch/val_scores_median': np.median(val_scores) if val_scores else 0.0,
            'epoch/val_scores_25th': np.percentile(val_scores, 25) if val_scores else 0.0,
            'epoch/val_scores_75th': np.percentile(val_scores, 75) if val_scores else 0.0
        })
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train Loss: {np.mean(train_losses) if train_losses else "nan":.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}')
        print(f'  Val Accuracy: {avg_val_accuracy:.4f}')
        print(f'  Test Loss: {avg_test_loss:.4f}')  # Added test loss print
        print(f'  Best Epoch: {best_epoch} (Val Loss: {best_val_loss:.4f})')
        print(f'  NaN Batches: {nan_batches}')
        print(f'  Learning Rate: {current_lr:.6f}')
        
        # Early stopping
        if patience_counter >= config.get('patience', 5):
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    print(f'Training completed! Best validation loss: {best_val_loss:.4f}')
    wandb.finish()
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    model.eval()
    test_losses = []
    test_accuracies = []
    
    with torch.no_grad():
        for batch in test_loader:
            query = batch['query'].to(device)
            pos_passage = batch['passage'].to(device)
            neg_passage = batch['passage'][torch.randperm(len(batch['passage']))].to(device)
            is_selected = batch['is_selected'].to(device)
            
            # Forward pass
            query_emb = model.query_tower(query)
            pos_emb = model.passage_tower(pos_passage)
            neg_emb = model.passage_tower(neg_passage)
            
            # Compute loss
            loss = triplet_loss_function(
                query_emb, pos_emb, neg_emb,
                margin=config.get('margin', 0.2)
            )
            test_losses.append(loss.item())
            
            # Calculate accuracy
            accuracy, _ = evaluate(model, query, pos_passage, is_selected)
            test_accuracies.append(accuracy)
    
    avg_test_loss = np.mean(test_losses)
    avg_test_accuracy = np.mean(test_accuracies)
    print(f"Final Test Loss: {avg_test_loss:.4f}")
    print(f"Final Test Accuracy: {avg_test_accuracy:.4f}")
    
    # Log final test metrics
    wandb.log({
        'final/test_loss': avg_test_loss,
        'final/test_accuracy': avg_test_accuracy
    })
    
    # Interactive example viewing with test set option
    print("\nWould you like to see examples from the test set? (y/n)")
    choice = input().lower()
    if choice == 'y':
        print("\nShowing test set examples:")
        show_random_example(model, test_loader, test_dataset, device, word_vectors)
    
    # Original validation set examples option
    print("\nWould you like to see examples from the validation set? (y/n)")
    while True:
        choice = input().lower()
        if choice == 'y':
            show_random_example(model, val_loader, val_dataset, device, word_vectors)
            print("\nWould you like to see another example? (y/n)")
        elif choice == 'n':
            break
        else:
            print("Please enter 'y' or 'n'")

if __name__ == '__main__':
    config = {
        'train_path': '/root/search-engine/data/msmarco/train.json',
        'val_path': '/root/search-engine/data/msmarco/val.json',
        'test_path': '/root/search-engine/data/msmarco/test.json',
        'word2vec_model_path': '/root/search-engine/models/text8_embeddings/word2vec_model',
        'vocab_size': 100000,
        'embedding_dim': 300,
        'hidden_dim': 256,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 10,
        'margin': 0.2,
        'patience': 5,
        'data_fraction': 2     #    Use 1/2 of the data by default
    }
    
    train_model(config)
