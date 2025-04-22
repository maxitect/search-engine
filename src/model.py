# src/model.py
import torch
import torch.nn as nn
from transformers import AutoModel

class TwoTowerModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased', embedding_dim=768):
        super().__init__()
        # Shared embedding layer (pretrained)
        self.embedding = AutoModel.from_pretrained(model_name)
        
        # Freeze embedding layer (optional)
        for param in self.embedding.parameters():
            param.requires_grad = False
        
        # Query encoder (RNN)
        self.query_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Document encoder (RNN)
        self.doc_encoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        
        # Projection heads
        self.query_proj = nn.Linear(512, 256)  # 2*256 for bidirectional
        self.doc_proj = nn.Linear(512, 256)

    def forward(self, query_input, doc_input):
        # Get embeddings - handle input format properly
        query_outputs = self.embedding(
            input_ids=query_input['input_ids'],
            attention_mask=query_input['attention_mask'],
            token_type_ids=query_input['token_type_ids'] if 'token_type_ids' in query_input else None
        )
        query_emb = query_outputs.last_hidden_state
        
        doc_outputs = self.embedding(
            input_ids=doc_input['input_ids'],
            attention_mask=doc_input['attention_mask'],
            token_type_ids=doc_input['token_type_ids'] if 'token_type_ids' in doc_input else None
        )
        doc_emb = doc_outputs.last_hidden_state
        
        # Encode sequences
        query_out, _ = self.query_encoder(query_emb)
        doc_out, _ = self.doc_encoder(doc_emb)
        
        # Use last hidden state as representation
        query_repr = query_out[:, -1, :]
        doc_repr = doc_out[:, -1, :]
        
        # Project to same space
        query_repr = self.query_proj(query_repr)
        doc_repr = self.doc_proj(doc_repr)
        
        return query_repr, doc_repr