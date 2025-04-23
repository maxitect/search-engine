import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        # Initialize embeddings
        self.in_embed.weight.data.uniform_(
            -0.5 / embedding_dim,
            0.5 / embedding_dim
        )
        self.out_embed.weight.data.uniform_(
            -0.5 / embedding_dim,
            0.5 / embedding_dim
        )

    def forward(self, input_word, output_word=None, neg_samples=None):
        # Get input embeddings
        input_embeds = self.in_embed(input_word).squeeze(1)

        if output_word is not None and neg_samples is not None:
            # Get positive embeddings
            pos_embeds = self.out_embed(output_word).squeeze(1)

            # Get negative embeddings
            neg_embeds = self.out_embed(neg_samples)

            return input_embeds, pos_embeds, neg_embeds

        return input_embeds


def negative_sampling_loss(input_embeds, pos_embeds, neg_embeds):
    # Positive samples: sigmoid(dot product of input and positive embeddings)
    pos_score = torch.sum(input_embeds * pos_embeds, dim=1)
    pos_loss = F.logsigmoid(pos_score)

    # For negative samples, we need to calculate
    # dot product between each input embedding
    # and all its corresponding negative samples

    batch_size, neg_samples, embed_dim = neg_embeds.shape

    # Reshape input_embeds to [batch_size, 1, embedding_dim] for broadcasting
    input_embeds = input_embeds.unsqueeze(1)

    # Calculate dot products for all negative samples at once
    # Result shape: [batch_size, neg_samples]
    neg_score = torch.sum(input_embeds * neg_embeds, dim=2)

    # Apply log sigmoid to negative scores (with negative sign)
    neg_loss = F.logsigmoid(-neg_score).sum(1)

    # Final loss: -(positive_loss + negative_loss)
    return -(pos_loss + neg_loss).mean()
