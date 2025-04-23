import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        self.in_embed.weight.data.uniform_(
            -0.5/embedding_dim, 0.5/embedding_dim
        )
        self.out_embed.weight.data.uniform_(
            -0.5/embedding_dim, 0.5/embedding_dim
        )

    def forward(self, input_word, output_word=None, neg_samples=None):
        input_embeds = self.in_embed(input_word)

        if output_word is not None and neg_samples is not None:
            pos_embeds = self.out_embed(output_word)
            neg_embeds = self.out_embed(neg_samples)

            return input_embeds, pos_embeds, neg_embeds

        return input_embeds


def negative_sampling_loss(input_embeds, pos_embeds, neg_embeds):
    # Positive samples: sigmoid(dot product of input and positive embeddings)
    pos_score = torch.sum(input_embeds * pos_embeds, dim=1)
    pos_loss = F.logsigmoid(pos_score)

    # Negative samples: sigmoid(-dot product of input and negative embeddings)
    # Shape: [batch_size, num_neg_samples, embedding_dim]
    neg_score = torch.bmm(neg_embeds, input_embeds.unsqueeze(2)).squeeze()
    neg_loss = F.logsigmoid(-neg_score).sum(1)

    # Final loss: -(pos_loss + neg_loss)
    return -(pos_loss + neg_loss).mean()
