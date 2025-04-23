import torch
import torch.nn as nn
import torch.nn.functional as F


class TowerBase(nn.Module):
    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_dim,
            output_dim,
            dropout_rate=0.2,
            skipgram_model=None,
            freeze_embeddings=True
    ):
        super().__init__()

        # Set up embedding layer (either from SkipGram or new)
        if skipgram_model is not None:
            self.embedding = skipgram_model.in_embed
            self.embedding.weight.requires_grad = not freeze_embeddings
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Bidirectional GRU for encoding
        self.bi_gru = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # Projection from GRU output to final embedding space
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        # Layer normalisation as specified in requirements
        self.layer_norm = nn.LayerNorm(output_dim)

        # Dropout for regularisation (within required 10-50% range)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x shape: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]

        # Process through bidirectional GRU
        # [batch_size, seq_len, hidden_dim*2]
        gru_output, _ = self.bi_gru(embedded)

        # Pool GRU outputs (mean pooling over sequence dimension)
        pooled = torch.mean(gru_output, dim=1)  # [batch_size, hidden_dim*2]

        # Project to output dimension
        output = self.fc(pooled)  # [batch_size, output_dim]

        # Apply layer normalisation
        output = self.layer_norm(output)

        # Apply dropout
        output = self.dropout(output)

        # Return L2 normalized output for cosine similarity
        return F.normalize(output, p=2, dim=1)


class QryTower(TowerBase):
    def __init__(
            self,
            vocab_size,
            embedding_dim=256,
            hidden_dim=256,
            output_dim=128,
            dropout_rate=0.2,
            skipgram_model=None,
            freeze_embeddings=True
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            skipgram_model=skipgram_model,
            freeze_embeddings=freeze_embeddings
        )


class DocTower(TowerBase):
    def __init__(
            self,
            vocab_size,
            embedding_dim=300,
            hidden_dim=256,
            output_dim=128,
            dropout_rate=0.2,
            skipgram_model=None,
            freeze_embeddings=True
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
            skipgram_model=skipgram_model,
            freeze_embeddings=freeze_embeddings
        )


class TwoTowerModel(nn.Module):
    def __init__(self, query_tower, doc_tower):
        super().__init__()
        self.query_tower = query_tower
        self.doc_tower = doc_tower

    def forward(self, query, document):
        query_embedding = self.query_tower(query)
        doc_embedding = self.doc_tower(document)
        return query_embedding, doc_embedding

    def compute_similarity(self, query_embedding, doc_embedding):
        return torch.sum(query_embedding * doc_embedding, dim=1)

    def batch_inference(self, query, document_list):
        query_embedding = self.query_tower(query)
        query_embedding = query_embedding.unsqueeze(
            1)  # [batch_size, 1, output_dim]

        all_similarities = []
        for doc_batch in document_list:
            doc_embedding = self.doc_tower(
                doc_batch)  # [batch_size, output_dim]
            doc_embedding = doc_embedding.unsqueeze(
                0)  # [1, batch_size, output_dim]

            # Compute similarity for all documents in batch
            # [batch_size, batch_size]
            similarity = torch.sum(query_embedding * doc_embedding, dim=2)
            all_similarities.append(similarity)

        # Concatenate all similarities
        all_similarities = torch.cat(
            all_similarities, dim=1)  # [batch_size, num_docs]

        # Get top 5 documents
        _, indices = torch.topk(all_similarities, k=5, dim=1)
        return indices


def cosine_similarity(query, document):
    query_norm = torch.norm(query, dim=1, keepdim=True)
    doc_norm = torch.norm(document, dim=1, keepdim=True)

    return torch.sum(
        query * document,
        dim=1
    ) / (query_norm * doc_norm).squeeze()


def triplet_loss_function(
    query,
    relevant_document,
    irrelevant_document,
    margin=0.3
):
    relevant_similarity = cosine_similarity(query, relevant_document)
    irrelevant_similarity = cosine_similarity(query, irrelevant_document)

    # Convert similarity to distance (1 - similarity)
    relevant_distance = 1 - relevant_similarity
    irrelevant_distance = 1 - irrelevant_similarity

    # Compute triplet loss with proper batching (element-wise maximum)
    triplet_loss = torch.clamp(
        relevant_distance - irrelevant_distance + margin, min=0)

    return triplet_loss.mean()
