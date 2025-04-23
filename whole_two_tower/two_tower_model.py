import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gensim.models import Word2Vec
import pickle

class EmbeddingMapper(nn.Module):
    """Maps words to Gensim embeddings and handles OOV words."""
    def __init__(self, gensim_model_path, embedding_dim=200):
        super().__init__()
        # Load Gensim model
        self.gensim_model = Word2Vec.load(gensim_model_path)
        self.embedding_dim = embedding_dim
        
        # Create embedding matrix
        self.vocab_size = len(self.gensim_model.wv)
        self.embedding_matrix = torch.FloatTensor(self.gensim_model.wv.vectors)
        
        # Initialize OOV embeddings
        self.oov_embedding = nn.Parameter(torch.randn(1, embedding_dim))
        
    def forward(self, word_indices):
        """Map word indices to embeddings, handling OOV words."""
        batch_size = word_indices.size(0)
        seq_len = word_indices.size(1)
        
        # Initialize output tensor
        embeddings = torch.zeros(batch_size, seq_len, self.embedding_dim)
        
        # Map known words to embeddings
        for i in range(batch_size):
            for j in range(seq_len):
                word_idx = word_indices[i, j].item()
                if word_idx < self.vocab_size:
                    embeddings[i, j] = self.embedding_matrix[word_idx]
                else:
                    embeddings[i, j] = self.oov_embedding
        
        return embeddings

class QueryTower(nn.Module):
    """BiRNN tower for processing queries."""
    def __init__(self, gensim_model_path, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding_mapper = EmbeddingMapper(gensim_model_path)
        self.hidden_dim = hidden_dim
        
        # Bidirectional RNN
        self.rnn = nn.GRU(
            input_size=self.embedding_mapper.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, query_indices):
        # Get embeddings
        embeddings = self.embedding_mapper(query_indices)
        
        # Process through RNN
        rnn_output, _ = self.rnn(embeddings)
        
        # Get final state (concatenate forward and backward)
        final_state = torch.cat([rnn_output[:, -1, :self.hidden_dim], 
                               rnn_output[:, 0, self.hidden_dim:]], dim=1)
        
        # Project to final representation
        output = self.projection(final_state)
        return output

class PassageTower(nn.Module):
    """Attention + RNN tower for processing passages."""
    def __init__(self, gensim_model_path, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding_mapper = EmbeddingMapper(gensim_model_path)
        self.hidden_dim = hidden_dim
        
        # RNN for passage encoding
        self.rnn = nn.GRU(
            input_size=self.embedding_mapper.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Output projection
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, passage_indices):
        # Get embeddings
        embeddings = self.embedding_mapper(passage_indices)
        
        # Process through RNN
        rnn_output, _ = self.rnn(embeddings)
        
        # Compute attention weights
        attention_weights = self.attention(rnn_output)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended = torch.sum(rnn_output * attention_weights, dim=1)
        
        # Project to final representation
        output = self.projection(attended)
        return output

class TwoTowerModel(nn.Module):
    """Combined model with query and passage towers."""
    def __init__(self, gensim_model_path, hidden_dim=256):
        super().__init__()
        self.query_tower = QueryTower(gensim_model_path, hidden_dim)
        self.passage_tower = PassageTower(gensim_model_path, hidden_dim)
        
    def forward(self, query_indices, passage_indices):
        # Get query and passage representations
        query_rep = self.query_tower(query_indices)
        passage_rep = self.passage_tower(passage_indices)
        
        # Compute similarity scores
        scores = torch.sum(query_rep * passage_rep, dim=1)
        return scores

def triplet_loss(query_rep, pos_passage_rep, neg_passage_rep, margin=0.3):
    """Compute triplet loss for query-positive-negative triplets."""
    pos_scores = torch.sum(query_rep * pos_passage_rep, dim=1)
    neg_scores = torch.sum(query_rep * neg_passage_rep, dim=1)
    
    loss = F.relu(margin - pos_scores + neg_scores)
    return loss.mean()

def train_step(model, optimizer, query_batch, pos_passage_batch, neg_passage_batch):
    """Perform a single training step."""
    model.train()
    optimizer.zero_grad()
    
    # Get representations
    query_rep = model.query_tower(query_batch)
    pos_rep = model.passage_tower(pos_passage_batch)
    neg_rep = model.passage_tower(neg_passage_batch)
    
    # Compute loss
    loss = triplet_loss(query_rep, pos_rep, neg_rep)
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate(model, query_batch, passage_batch, is_selected):
    """Evaluate model on a batch of data."""
    model.eval()
    with torch.no_grad():
        # Get scores for all query-passage pairs
        scores = model(query_batch, passage_batch)
        
        # Compute metrics
        predictions = torch.argmax(scores, dim=1)
        correct = torch.sum(predictions == torch.argmax(is_selected, dim=1))
        accuracy = correct.float() / len(is_selected)
        
        return accuracy.item() 