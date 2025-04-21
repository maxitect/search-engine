from __future__ import annotations

import glob

import torch
import torch.nn as nn
import wandb

from .tokeniser import get_tokeniser

# from utils import get_device


class Word2Vec(torch.nn.Module):
    def __init__(
        self, vocab_size: int, embedding_dim: int,
        mode: str = 'skipgram',
    ):
        """
        Initialize Word2Vec model.

        CBOW:
        Mikolov, Tomas, Kai Chen, Greg Corrado, and Jeffrey Dean.
        ‘Efficient Estimation of Word Representations in Vector Space’.
        arXiv, 7 September 2013. https://doi.org/10.48550/arXiv.1301.3781.

        Skipgram:
        Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean.
        ‘Distributed Representations of Words and Phrases and Their
        Compositionality’. In Advances in Neural Information Processing Systems,
        Vol. 26. Curran Associates, Inc., 2013.


        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            architecture: Either 'skipgram' or 'cbow'
        """
        super().__init__()

        # self.vocab_size = vocab_size
        # self.embedding_dim = embedding_dim
        self.architecture = mode.lower()

        # Initialize embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Initialise final projection layer
        self.fc_output = nn.Linear(embedding_dim, vocab_size)
        # self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, input_words: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_words (torch.Tensor): Tensor of input word indices
                (batch_size, N_words )

        Returns:
            scores (torch.Tensor): raw output scores from the final projection
                layer (batch_size, vocab_size)
        """
        if self.architecture == 'skipgram':
            # Skip-gram: predict context words from center word
            input_vectors = self.embeddings(input_words)
            scores = self.fc_output(input_vectors)

        else:
            # CBOW: predict center word from context words
            input_vectors = self.embeddings(input_words)
            # Average over the context words
            context_vector = torch.mean(input_vectors, dim=-2)
            scores = self.fc_output(context_vector)

        return scores

    def get_word_embedding(self, word_idx: int) -> torch.Tensor:
        """Get the embedding for a specific word."""
        return self.embeddings(torch.tensor([word_idx]))


class SkipGram(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        """
        Initialize Word2Vec SkipGram model.

        Skipgram:
        Mikolov, Tomas, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean.
        ‘Distributed Representations of Words and Phrases and Their
        Compositionality’. In Advances in Neural Information Processing Systems,
        Vol. 26. Curran Associates, Inc., 2013.

        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
        """
        super().__init__()

        # Initialize embeddings
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def forward(self, input_words: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_words (torch.Tensor): Tensor of input word indices
                (batch_size, N_words )

        Returns:
            scores (torch.Tensor): raw output scores from the final projection
                layer (batch_size, vocab_size)
        """
        # Skip-gram: predict context words from center word
        embedded_vectors = self.embeddings(input_words)
        return embedded_vectors

    def get_word_embedding(self, word_idx: int) -> torch.Tensor:
        """Get the embedding for a specific word."""
        return self.embeddings(torch.tensor([word_idx]))


def skipgram_loss(
    positive_embeddings: torch.Tensor,
    center_embeddings: torch.Tensor,
    negative_embeddings: torch.Tensor,
) -> float:
    """ Compute the loss for the skipgram model.

    num_negative_samples = 2*C*K
    Args:
        positive_embeddings: (B, 2C, embedding_dim)
        center_embeddings: (B, embedding_dim)
        negative_embeddings: (B, num_negative_samples, embedding_dim)

    Returns:
        loss: some float
    """

    B = positive_embeddings.shape[0]
    log_sigmoid = nn.LogSigmoid()

    # More positive is better as we are doing dot product with log sigmoid
    pos = log_sigmoid(
        torch.einsum(
            'b ij, bj -> b', positive_embeddings, center_embeddings,
        ),
    )
    neg = log_sigmoid(
        -torch.einsum(
            'b ij, bj -> b',
            negative_embeddings, center_embeddings,
        ),
    )
    # Average over the batch
    # loss = -(pos.mean() + neg.mean()) / B
    loss = -(pos.sum() + neg.sum()) / B
    return loss


def get_word2vec_from_checkpoint(checkpoint_path: str):
    """
    Get the word2vec model from the checkpoint
    """
    # device = get_device()
    checkpoint = torch.load(checkpoint_path)

    embedding_dim = checkpoint['embedding_dim']
    vocab_size = checkpoint['vocab_size']
    mode = checkpoint['mode']
    if mode == 'cbow':
        model = Word2Vec(vocab_size, embedding_dim, mode='cbow')
    else:
        model = SkipGram(vocab_size, embedding_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def setup_word2vec():
    text8_path = hf_hub_download(
        repo_id='kwokkenton/hn-upvotes', filename='text8.parquet', repo_type='dataset')

    # Load the word2vec model
    w2v_checkpoint_path = get_wandb_checkpoint_path(
        'kwokkenton-individual/mlx-week1-word2vec/skipgram:v34',
    )

    tokeniser = get_tokeniser(text8_path)
    # vocab_size = tokeniser.vocab_size
    w2v_model = get_word2vec_from_checkpoint(w2v_checkpoint_path).eval()

    return tokeniser, w2v_model
