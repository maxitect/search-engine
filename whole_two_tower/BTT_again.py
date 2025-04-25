import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import pickle
from transformers import BertTokenizer
import os
from tqdm import tqdm

class MSMARCODataset(Dataset):
    def __init__(self, data_path, max_query_len=32, max_passage_len=256, data_fraction=1.0):
        self.data = []
        self.max_query_len = max_query_len
        self.max_passage_len = max_passage_len
        
        print("Loading BERT tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        print("Loading BERT embeddings...")
        with open('/root/search-engine/models/text8_embeddings/inherited_bert_embeddings.pkl', 'rb') as f:
            self.embeddings = pickle.load(f)
        
        print("Loading vocabulary mapping...")
        with open('/root/search-engine/models/text8_embeddings/inherited_bert_vocab.json', 'r') as f:
            vocab_list = json.load(f)
            self.vocab = {word: idx for idx, word in enumerate(vocab_list)}
            self.reverse_vocab = {str(idx): word for idx, word in enumerate(vocab_list)}
        
        print(f"Loading and preprocessing data from {data_path}...")
        with open(data_path, 'r') as f:
            total_lines = sum(1 for _ in f)
        
        with open(data_path, 'r') as f:
            for i, line in enumerate(tqdm(f, total=total_lines, desc="Processing data")):
                item = json.loads(line)
                query_text = item['query']
                passages = item['passages']
                is_selected = item['is_selected']
                
                for passage, selected in zip(passages, is_selected):
                    passage_text = passage['passage_text']
                    
                    # Store original text and tokenized version
                    self.data.append({
                        'query_text': query_text,
                        'passage_text': passage_text,
                        'is_selected': selected,
                        'query': self.tokenizer.encode(
                            query_text,
                            add_special_tokens=True,
                            max_length=max_query_len,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        ).squeeze(0),
                        'passage': self.tokenizer.encode(
                            passage_text,
                            add_special_tokens=True,
                            max_length=max_passage_len,
                            padding='max_length',
                            truncation=True,
                            return_tensors='pt'
                        ).squeeze(0)
                    })
        
        # Use configurable fraction of the data
        fraction = 1.0 / data_fraction
        self.data = self.data[:int(len(self.data) * fraction)]
        print(f"Dataset loaded with {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'query': item['query'],
            'passage': item['passage'],
            'is_selected': torch.FloatTensor([item['is_selected']])
        }
    
    def get_text(self, idx):
        """Get the original text for a given index."""
        item = self.data[idx]
        return {
            'query_text': item['query_text'],
            'passage_text': item['passage_text'],
            'is_selected': item['is_selected']
        }

class TwoTowerModel(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=256):
        super().__init__()
        
        # Query tower
        self.query_embedding = nn.Embedding(30522, embedding_dim)  # BERT vocab size
        self.query_gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.query_projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Passage tower
        self.passage_embedding = nn.Embedding(30522, embedding_dim)  # BERT vocab size
        self.passage_gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
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
        query_emb = self.query_embedding(query)
        query_rnn_out, _ = self.query_gru(query_emb)
        query_rep = self.query_projection(query_rnn_out[:, -1, :])
        return query_rep
    
    def passage_tower(self, passage):
        """Process passage through passage tower."""
        passage_emb = self.passage_embedding(passage)
        passage_rnn_out, _ = self.passage_gru(passage_emb)
        passage_rep = self.passage_projection(passage_rnn_out[:, -1, :])
        return passage_rep
    
    def forward(self, query, passage):
        query_rep = self.query_tower(query)
        passage_rep = self.passage_tower(passage)
        return torch.sum(query_rep * passage_rep, dim=1)

def print_top_10_passages(model, test_dataset):
    """Print the top 6 most similar passages for a random query from the test set."""
    device = next(model.parameters()).device
    
    # Pick a random query from the test set
    random_idx = np.random.randint(0, len(test_dataset))
    item = test_dataset.get_text(random_idx)
    
    # Get the original query text
    query_text = item['query_text']
    print(f"\nQuery: {query_text}")
    
    # Get the tokenized version for the model
    query_tokens = test_dataset.tokenizer.encode(
        query_text,
        add_special_tokens=True,
        max_length=test_dataset.max_query_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).squeeze(0).to(device)
    
    # Get query embedding
    with torch.no_grad():
        query_embedding = model.query_tower(query_tokens.unsqueeze(0))
        
        print("Computing similarities with all passages...")
        # Compute similarities with all passages in test set
        similarities = []
        for idx in tqdm(range(len(test_dataset)), desc="Processing passages"):
            # Get the tokenized version for the model
            item_tokens = test_dataset[idx]
            passage_tokens = item_tokens['passage'].to(device)
            passage_embedding = model.passage_tower(passage_tokens.unsqueeze(0))
            similarity = torch.nn.functional.cosine_similarity(
                query_embedding, passage_embedding, dim=1
            ).item()
            similarities.append((idx, similarity))
        
        # Sort by similarity and get top 6
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:6]
        
        print("\nTop 6 results:")
        for idx, score in top_results:
            # Get the original text using get_text
            item = test_dataset.get_text(idx)
            print(f"\nSimilarity Score: {score:.4f}")
            print("Content:")
            print("-" * 80)
            print(item['passage_text'])
            print("-" * 80)

def train_model(config):
    print("Initializing training...")
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\nLoading training dataset...")
    train_dataset = MSMARCODataset(
        config['train_path'],
        data_fraction=config['data_fraction']
    )
    
    print("\nLoading validation dataset...")
    val_dataset = MSMARCODataset(
        config['val_path'],
        data_fraction=config['data_fraction']
    )
    
    print("\nLoading test dataset...")
    test_dataset = MSMARCODataset(
        config['test_path'],
        data_fraction=config['data_fraction']
    )
    
    print("\nCreating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    print("\nInitializing model...")
    model = TwoTowerModel(
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim']
    )
    model = model.to(device)
    
    # Print initial top passages before training
    print("\nInitial Top Passages (Before Training):")
    print("=" * 80)
    print_top_10_passages(model, test_dataset)
    print("=" * 80)
    
    print("\nInitializing optimizer...")
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    print("\nStarting training...")
    # Training loop
    for epoch in range(config['epochs']):
        model.train()
        train_losses = []
        
        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        for batch in progress:
            query = batch['query'].to(device)
            pos_passage = batch['passage'].to(device)
            neg_passage = batch['passage'][torch.randperm(len(batch['passage']))].to(device)
            
            # Forward pass
            query_emb = model.query_tower(query)
            pos_emb = model.passage_tower(pos_passage)
            neg_emb = model.passage_tower(neg_passage)
            
            # Compute triplet loss
            pos_scores = torch.sum(query_emb * pos_emb, dim=1)
            neg_scores = torch.sum(query_emb * neg_emb, dim=1)
            loss = torch.clamp(neg_scores - pos_scores + config['margin'], min=0.0).mean()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            progress.set_postfix({'loss': loss.item()})
        
        # Print top passages after each epoch
        print("\nTop passages after epoch", epoch + 1)
        print_top_10_passages(model, test_dataset)
        
        print(f'Epoch {epoch+1}:')
        print(f'  Train Loss: {np.mean(train_losses):.4f}')

if __name__ == '__main__':
    config = {
        'train_path': '/root/search-engine/data/msmarco/train.json',
        'val_path': '/root/search-engine/data/msmarco/val.json',
        'test_path': '/root/search-engine/data/msmarco/test.json',
        'embedding_dim': 768,
        'hidden_dim': 256,
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 10,
        'margin': 0.2,
        'data_fraction': 99
    }
    
    train_model(config)
