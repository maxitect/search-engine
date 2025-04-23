import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gensim.models import Word2Vec
import pickle

class EmbeddingMapper(nn.Module):
    """Maps words to Gensim embeddings and handles OOV words."""
    def __init__(self, gensim_model_path):
        super().__init__()
        # Load Gensim model
        self.gensim_model = Word2Vec.load(gensim_model_path)
        self.embedding_dim = self.gensim_model.vector_size
        self.vocab_size = len(self.gensim_model.wv)
        
        # Create embedding layer
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        # Initialize embeddings with Gensim weights
        with torch.no_grad():
            for i in range(self.vocab_size):
                word = self.gensim_model.wv.index_to_key[i]
                self.embedding.weight[i] = torch.FloatTensor(self.gensim_model.wv[word])
    
    def forward(self, x):
        return self.embedding(x)

class QueryTower(nn.Module):
    """BiRNN tower for processing queries."""
    def __init__(self, gensim_model_path, hidden_dim):
        super().__init__()
        self.embedding_mapper = EmbeddingMapper(gensim_model_path)
        self.rnn = nn.GRU(
            input_size=self.embedding_mapper.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embeddings = self.embedding_mapper(x)
        # embeddings shape: (batch_size, seq_len, embedding_dim)
        rnn_output, _ = self.rnn(embeddings)
        # rnn_output shape: (batch_size, seq_len, hidden_dim * 2)
        query_rep = self.projection(rnn_output[:, -1, :])
        # query_rep shape: (batch_size, hidden_dim)
        return query_rep

class PassageTower(nn.Module):
    """Attention + RNN tower for processing passages."""
    def __init__(self, gensim_model_path, hidden_dim):
        super().__init__()
        self.embedding_mapper = EmbeddingMapper(gensim_model_path)
        self.rnn = nn.GRU(
            input_size=self.embedding_mapper.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embeddings = self.embedding_mapper(x)
        # embeddings shape: (batch_size, seq_len, embedding_dim)
        rnn_output, _ = self.rnn(embeddings)
        # rnn_output shape: (batch_size, seq_len, hidden_dim * 2)
        
        # Compute attention weights
        attention_weights = self.attention(rnn_output)
        # attention_weights shape: (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply attention
        attended = torch.sum(rnn_output * attention_weights, dim=1)
        # attended shape: (batch_size, hidden_dim * 2)
        
        passage_rep = self.projection(attended)
        # passage_rep shape: (batch_size, hidden_dim)
        return passage_rep

class TwoTowerModel(nn.Module):
    """Combined model with query and passage towers."""
    def __init__(self, gensim_model_path, hidden_dim):
        super().__init__()
        self.query_tower = QueryTower(gensim_model_path, hidden_dim)
        self.passage_tower = PassageTower(gensim_model_path, hidden_dim)
    
    def forward(self, query, passage):
        query_rep = self.query_tower(query)
        passage_rep = self.passage_tower(passage)
        return torch.sum(query_rep * passage_rep, dim=1)

def train_step(model, optimizer, query_batch, pos_passage_batch, neg_passage_batch):
    # Forward pass
    pos_scores = model(query_batch, pos_passage_batch)
    neg_scores = model(query_batch, neg_passage_batch)
    
    # Compute loss
    loss = -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores)))
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

def evaluate(model, query_batch, passage_batch, labels):
    with torch.no_grad():
        scores = model(query_batch, passage_batch)
        predictions = (scores > 0).float()
        accuracy = (predictions == labels).float().mean()
    return accuracy.item() 